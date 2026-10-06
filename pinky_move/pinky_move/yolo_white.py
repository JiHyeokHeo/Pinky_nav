"""시뮬 실험: YOLO로 확인한 차선만 현재 흰 픽셀로 연결/추적한다."""
import cv2
import numpy as np
from .white_lane import white_floor_masks


class YoloWhiteSupplement:
    """흰색만으로 차선을 새로 만들지 않는 semantic-seeded 보충기.

    현재 YOLO 마스크와 겹치는 흰 연결 성분을 우선 사용한다. YOLO가 잠시
    사라지면 이전 선택 성분을 현재 이미지로 optical-flow warp해서 대응한다.
    결과는 항상 CURRENT 흰 성분이며 과거 마스크를 그대로 발행하지 않는다.
    마지막 YOLO 확인 10초, 영상 간격 0.8초를 넘으면 해당 성분은 사용할 수 없다.
    """
    def __init__(self, semantic_timeout_s=10., frame_gap_s=.8, support_interval=None):
        self.semantic_timeout_s = semantic_timeout_s
        self.frame_gap_s = frame_gap_s
        self.previous_gray = None
        self.previous_at = None
        self.tracks = []
        self.support_interval = support_interval

    def update(self, frame, lane_masks, now_s, calibration, white_masks=None):
        if not np.isfinite(now_s):
            raise ValueError('invalid supplementation timestamp')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        components = white_floor_masks(frame, calibration) if white_masks is None else white_masks
        # A white floor or broad pale object must not become an entire lane.
        components = [m for m in components if m.shape == gray.shape and
                      60 <= np.count_nonzero(m) <= .18*height*width]
        unsupported = 0
        if self.support_interval is not None:
            # 먼 S자 출구 조각은 현재 제어 구간의 경계가 아니다. 이미지 높이가
            # 아니라 보정된 바닥 좌표로 검사하고, 가까운 부분이 있는 긴 선은
            # 통째로 보존한다. 과거 선을 이어 그리거나 연결하지 않는다.
            from .robot_projection import robot_floor_points
            lo, hi = self.support_interval
            supported = []
            for component in components:
                rows, cols = np.nonzero(component)
                pixels = np.column_stack((cols[::4], rows[::4]))
                points = robot_floor_points(pixels, calibration, (width, height), 2.)
                valid = np.isfinite(points).all(axis=1) & (points[:, 0] >= lo) & (points[:, 0] <= hi)
                if np.count_nonzero(valid) >= 5:
                    supported.append(component)
                else:
                    unsupported += 1
            components = supported
        masks = [(m > 0).astype(np.uint8) for m in lane_masks]
        # 하나의 semantic instance가 여러 독립 흰 조각을 모두 경계로
        # 증식시키지 않도록 instance별 하나의 유일한 대응만 완성한다.
        current_seed = set()
        completed_instances = set()
        for instance, mask in enumerate(masks):
            seed = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            candidates = []
            for index, component in enumerate(components):
                overlap = np.count_nonzero(component & seed)
                area = np.count_nonzero(component)
                if overlap >= max(20, .05*area):
                    score = 2*overlap/max(1, area+np.count_nonzero(seed))
                    candidates.append((score, index))
            candidates.sort(reverse=True)
            if candidates and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] >= .05):
                current_seed.add(candidates[0][1])
                completed_instances.add(instance)
        warped = []
        if (self.previous_gray is not None and self.previous_gray.shape == gray.shape and
                self.previous_at is not None and 0 < now_s-self.previous_at <= self.frame_gap_s):
            # Backward flow maps each current pixel to its previous-frame origin.
            shape = (max(1, width//2), max(1, height//2))
            flow = cv2.calcOpticalFlowFarneback(cv2.resize(gray, shape),
                cv2.resize(self.previous_gray, shape), None, .5, 3, 15, 3, 5, 1.2, 0)
            flow = cv2.resize(flow, (width, height))
            flow[:, :, 0] *= width/shape[0]
            flow[:, :, 1] *= height/shape[1]
            x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            for mask, last_semantic in self.tracks:
                if 0 <= now_s-last_semantic <= self.semantic_timeout_s:
                    mapped = cv2.remap(mask, x+flow[:, :, 0], y+flow[:, :, 1],
                                       cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                    warped.append((mapped, last_semantic))
        selected, next_tracks, history_count = [], [], 0
        # Optical-flow도 양방향 유일 대응을 요구한다. 한 과거 선을 여러
        # 조각으로 복제하지 않고, 같은 선의 현재 semantic 재확인을 우선한다.
        scores = np.zeros((len(components), len(warped)))
        for index, component in enumerate(components):
            for old, (previous, _) in enumerate(warped):
                scores[index, old] = np.count_nonzero(component & previous)/max(
                    1, max(np.count_nonzero(component), np.count_nonzero(previous)))
        consumed = set()
        for index in current_seed:
            if len(warped) and scores[index].max() >= .35:
                consumed.add(int(scores[index].argmax()))
        for index, mask in enumerate(components):
            semantic_at = now_s if index in current_seed else None
            if semantic_at is None:
                candidates = []
                for old, (_, timestamp) in enumerate(warped):
                    if old in consumed or scores[index, old] < .35:
                        continue
                    competing = [scores[other, old] for other in range(len(components)) if other != index]
                    if competing and scores[index, old]-max(competing) < .1:
                        continue
                    candidates.append((scores[index, old], timestamp, old))
                candidates.sort(reverse=True)
                if candidates and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] >= .1):
                    semantic_at = candidates[0][1]
                    consumed.add(candidates[0][2])
                    history_count += 1
            if semantic_at is not None:
                selected.append(mask)
                next_tracks.append((mask.copy(), semantic_at))
        # Keep uncompleted YOLO observations, including nonwhite genuine lanes.
        retained = [mask for instance, mask in enumerate(masks) if instance not in completed_instances and not any(
            np.count_nonzero(mask & complete) >= max(10, .15*np.count_nonzero(mask))
            for complete in selected)]
        self.previous_gray, self.previous_at, self.tracks = gray, now_s, next_tracks
        return selected+retained, dict(yolo=len(masks), white_current=len(selected)-history_count,
            white_tracked=history_count, retained_yolo=len(retained),
            semantic_timeout_s=self.semantic_timeout_s, unsupported_white=unsupported)
