"""Read-only mission diagnostics shared by the server and map UI."""
import json
import math

VIEW_TOPIC = '/central_control/mission_view'
ROBOTS = ('pinky1', 'pinky2')


def mission_snapshot(server):
    """Expose the checked plans, not a second UI-side planning algorithm."""
    states = dict(server._task_states)
    for robot in server._mission_robots:
        if robot in server._completed_robots:
            states[robot] = 'completed'
        elif server._safety_fault:
            states[robot] = 'stopped'
        elif getattr(server, '_detour_robot', None) is not None:
            states[robot] = 'detouring' if robot == server._detour_robot else 'holding'
        elif server._route_refresh_pending:
            states[robot] = 'replanning'
        elif robot in (server._yielding_robot, server._reserved_robot):
            states[robot] = 'yielding'
        elif robot in server._active_nav_goals:
            states[robot] = 'navigating'
        else:
            states.setdefault(robot, 'planning')
    priority = getattr(server, '_effective_priority_robot', None) or server.get_parameter('priority_robot').value.strip('/')
    if server._yielding_robot == priority:
        priority = next(robot for robot in ROBOTS if robot != server._yielding_robot)
    return dict(version=1, generation=server._mission_generation,
                active=server._mission_active, state=server._diagnostic_state,
                reason=server._safety_fault or server._mission_view_reason,
                priority=priority,
                yielding=server._yielding_robot or server._reserved_robot,
                yield_points=[[p.pose.position.x,p.pose.position.y] for p in getattr(server, '_yield_points', [])],
                detour_point=getattr(server, '_detour_point', None),
                robots=states, paths={robot: dict(frame_id=server._display_frames[robot],
                    points=points) for robot, points in server._display_paths.items()})


def parse_view(data):
    """Bound untrusted visualization data; never pass NaNs into Qt painting."""
    if len(data) > 2_000_000:
        raise ValueError('mission view too large')
    view = json.loads(data)
    if not isinstance(view, dict) or view.get('version') != 1:
        raise ValueError('unsupported mission view')
    if type(view.get('generation')) is not int or type(view.get('active')) is not bool:
        raise ValueError('invalid mission identity')
    for key in ('state', 'reason', 'priority'):
        if not isinstance(view.get(key), str) or len(view[key]) > 2000:
            raise ValueError('invalid mission status')
    if view.get('yielding') not in (*ROBOTS, None):
        raise ValueError('invalid yielding robot')
    paths, states = view.get('paths'), view.get('robots')
    if not isinstance(paths, dict) or not isinstance(states, dict):
        raise ValueError('invalid mission routes')
    if any(robot not in ROBOTS for robot in (*paths, *states)):
        raise ValueError('unknown robot')
    if any(not isinstance(state, str) or len(state) > 100 for state in states.values()):
        raise ValueError('invalid robot state')
    for path in paths.values():
        if not isinstance(path, dict) or not isinstance(path.get('frame_id'), str):
            raise ValueError('invalid path frame')
        points = path.get('points')
        if not isinstance(points, list) or not 1 <= len(points) <= 20000:
            raise ValueError('invalid path points')
        for point in points:
            if (not isinstance(point, list) or len(point) != 2 or
                    any(type(v) not in (float, int) or not math.isfinite(v) for v in point)):
                raise ValueError('nonfinite path coordinate')
    points = view.get('yield_points', [])
    if not isinstance(points, list) or len(points) > 32:
        raise ValueError('invalid yield points')
    for point in points + ([view['detour_point']] if view.get('detour_point') is not None else []):
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(v) not in (float,int) or not math.isfinite(v) for v in point)):
            raise ValueError('invalid yield-point coordinate')
    return view


def visible_paths(view, frame):
    """Do not draw a path in a different physical map coordinate system."""
    return {robot: path['points'] for robot, path in view['paths'].items()
            if path['frame_id'] == frame}


def status_text(view):
    labels = {'idle': '대기', 'planning': '경로 계산', 'navigating': '주행 중',
              'yielding': '양보 대기', 'replanning': '경로 재계산', 'completed': '도착 완료',
              'parked': '주차 완료', 'patrol_complete': '순찰 완료', 'stopped': '정지',
              'failed': '실패', 'canceled': '취소', 'succeeded': '전체 완료'}
    labels.update(detouring='대피점 이동', holding='상대 대피 중 정지')
    robots = ' | '.join(f'{robot}: {labels.get(view["robots"].get(robot, "idle"), view["robots"].get(robot))}'
                        for robot in ROBOTS)
    return f'{robots}\n경합 검사: {labels.get(view["state"], view["state"])} · 우선 {view["priority"]}\n{view["reason"]}'
