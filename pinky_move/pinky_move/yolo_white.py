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
    def __init__(self, semantic_timeout_s=10., frame_gap_s=.8):
        self.semantic_timeout_s = semantic_timeout_s
        self.frame_gap_s = frame_gap_s
        self.previous_gray = None
        self.previous_at = None
        self.tracks = []

    def update(self, frame, lane_masks, now_s, calibration, white_masks=None):
        if not np.isfinite(now_s):
            raise ValueError('invalid supplementation timestamp')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        components = white_floor_masks(frame, calibration) if white_masks is None else white_masks
        # A white floor or broad pale object must not become an entire lane.
        components = [m for m in components if m.shape == gray.shape and
                      60 <= np.count_nonzero(m) <= .18*height*width]
        masks = [(m > 0).astype(np.uint8) for m in lane_masks]
        union = np.maximum.reduce(masks) if masks else np.zeros(gray.shape, np.uint8)
        seed = cv2.dilate(union, np.ones((9, 9), np.uint8))
        current_seed = [np.count_nonzero(m & seed) >= max(20, .05*np.count_nonzero(m))
                        for m in components]
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
        for index, mask in enumerate(components):
            semantic_at = now_s if current_seed[index] else None
            if semantic_at is None:
                candidates = []
                for previous, timestamp in warped:
                    overlap = np.count_nonzero(mask & previous)/max(
                        1, min(np.count_nonzero(mask), np.count_nonzero(previous)))
                    if overlap >= .35:
                        candidates.append((overlap, timestamp))
                candidates.sort(reverse=True)
                if candidates and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] >= .1):
                    semantic_at = candidates[0][1]
                    history_count += 1
            if semantic_at is not None:
                selected.append(mask)
                next_tracks.append((mask.copy(), semantic_at))
        # Keep uncompleted YOLO observations, including nonwhite genuine lanes.
        retained = [mask for mask in masks if not any(
            np.count_nonzero(mask & complete) >= max(10, .15*np.count_nonzero(mask))
            for complete in selected)]
        self.previous_gray, self.previous_at, self.tracks = gray, now_s, next_tracks
        return selected+retained, dict(yolo=len(masks), white_current=len(selected)-history_count,
            white_tracked=history_count, retained_yolo=len(retained),
            semantic_timeout_s=self.semantic_timeout_s)
