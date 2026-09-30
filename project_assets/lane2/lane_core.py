#!/usr/bin/env python3
"""차선 추종의 판단 로직. ROS 도 카메라도 없이 단독으로 시험할 수 있게 분리했다.

여기 있는 함수들은 모두 순수 계산이다. 실제 주행 노드(lane_follow.py)는
이것들을 가져다 쓰기만 한다.
"""

import math

# 차선을 놓쳤을 때의 동작 단계
HOLD, SEARCH, STOP = 'HOLD', 'SEARCH', 'STOP'


def assign_sides(xs, prev_left, prev_right, lane_width, center):
    """검출된 차선 x 좌표들을 왼쪽/오른쪽에 배정한다.

    한 장의 사진만으로는 어느 쪽 차선인지 알 수 없다(양옆이 똑같은 카펫이다).
    그래서 '직전 프레임에서 가까운 쪽' 이라는 연속성으로 판단한다.
    차선은 순간이동하지 않으므로 급회전으로 왼쪽 차선이 화면 오른쪽까지
    밀려가도 계속 따라붙는다.

    반환: (left_x, right_x)  - 모르면 None
    """
    xs = sorted(xs)

    if not xs:
        return None, None

    if len(xs) >= 2:
        # 둘 이상이면 가장 멀리 떨어진 한 쌍을 차선으로 본다
        return xs[0], xs[-1]

    x = xs[0]

    # 직전에 본 위치가 있으면 그쪽에 잇는다
    if prev_left is not None and prev_right is not None:
        if abs(x - prev_left) <= abs(x - prev_right):
            return x, None
        return None, x
    if prev_left is not None:
        return (x, None) if abs(x - prev_left) < lane_width * 0.7 else (None, x)
    if prev_right is not None:
        return (None, x) if abs(x - prev_right) < lane_width * 0.7 else (x, None)

    # 아무 기억도 없으면(시작 직후) 화면 중앙 기준으로 가른다
    return (x, None) if x < center else (None, x)


def target_x(left, right, lane_width, center):
    """가야 할 지점의 x 좌표. 한쪽만 보이면 차선폭 절반만큼 밀어서 만든다."""
    if left is not None and right is not None:
        return (left + right) / 2.0
    if left is not None:
        return left + lane_width / 2.0
    if right is not None:
        return right - lane_width / 2.0
    return None


def steering(target, center, half_width, kp, kd, prev_error, dt, max_angular):
    """목표 x 로부터 각속도를 낸다. 오차는 화면 폭으로 정규화한다."""
    error = (target - center) / half_width          # -1 ~ +1
    derivative = (error - prev_error) / dt if dt > 0 and prev_error is not None else 0.0

    # 목표가 오른쪽(+)이면 오른쪽으로 = 음의 angular.z
    angular = -(kp * error + kd * derivative)
    angular = max(-max_angular, min(max_angular, angular))
    return angular, error


def linear_speed(error, base_speed, min_speed):
    """많이 틀어져 있으면 속도를 줄여 코너를 안정적으로 돈다."""
    scale = max(0.0, 1.0 - abs(error))
    return max(min_speed, base_speed * scale)


def lost_action(lost_sec, hold_sec, search_sec):
    """차선을 놓친 지 lost_sec 초 지났을 때 무엇을 할지."""
    if lost_sec < hold_sec:
        return HOLD        # 짧은 끊김은 직전 조향을 유지하며 지나간다
    if lost_sec < hold_sec + search_sec:
        return SEARCH      # 마지막으로 본 방향으로 천천히 돌며 찾는다
    return STOP


def search_direction(last_left, last_right, center):
    """탐색 회전 방향. 마지막으로 본 차선 쪽으로 돈다. +1 이면 왼쪽(반시계)."""
    if last_left is not None and last_right is None:
        return 1.0         # 왼쪽 차선만 봤다 -> 왼쪽으로 틀면 다시 잡힌다
    if last_right is not None and last_left is None:
        return -1.0
    if last_left is not None and last_right is not None:
        middle = (last_left + last_right) / 2.0
        return 1.0 if middle < center else -1.0
    return 1.0
