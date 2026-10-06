"""No ROS initialization: registered-point selection, evacuation, fences."""
import asyncio
import time
from types import SimpleNamespace as NS, MethodType
from unittest.mock import AsyncMock, Mock
import pytest
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from action_msgs.msg import GoalStatus
from rclpy.action import GoalResponse
from pinky_multi_robot.mission_action_server import MissionActionServer
from pinky_multi_robot.yield_detour import YieldDetourMixin
from pinky_interfaces.msg import RobotTask


def goal(x,y):
    p=PoseStamped(); p.header.frame_id='map'
    p.pose.position.x,p.pose.position.y=float(x),float(y)
    p.pose.orientation.w=1.
    return p


def server():
    poses={}
    for robot,xy in [('pinky1',(0.,0.)),('pinky2',(1.,0.))]:
        p=PoseWithCovarianceStamped(); p.header.frame_id='map'
        p.pose.pose=goal(*xy).pose
        poses[robot]=p
    params=dict(path_clearance=.35,conflict_distance=.55,planning_timeout=15.,
                yield_detour_timeout=60.,yield_arrival_tolerance=.15,
                pose_max_age=10.,priority_robot='pinky1',allow_namespaced_map_frames=False)
    async def compute(_robot,start,end):
        return [(start.pose.position.x,start.pose.position.y),(end.pose.position.x,end.pose.position.y)]
    s=NS(_poses=poses,_pose_receipt={r:time.monotonic() for r in poses},
         _yield_points=[goal(1.,1.)],_mission_goal=NS(is_cancel_requested=False),
         _mission_generation=1,_mission_active=True,_safety_fault=None,
         _detour_attempted=set(),_detour_robot=None,_detour_point=None,
         _effective_priority_robot=None,_display_paths={},_display_frames={},
         _active_nav_goals={},_nav_targets={},_completed_robots=set(),
         _route_refresh_pending=False,_detour_work_active=False,
         _compute_path=AsyncMock(side_effect=compute),
         get_parameter=lambda name:NS(value=params[name]),get_logger=lambda:Mock(),
         _sleep=AsyncMock(),_publish_mission_view=Mock())
    s.params=params
    for name in ('_fresh_start','_select_yield_point','_evacuate','_detour_and_resume'):
        setattr(s,name,MethodType(getattr(YieldDetourMixin,name),s))
    for name in ('_map_frame','_preflight'):
        setattr(s,name,MethodType(getattr(MissionActionServer,name),s))
    return s


def task(robot,x,y):
    t=RobotTask(); t.robot_id=robot; t.task_type=RobotTask.PARK
    t.parking_goal=goal(x,y)
    return t


def test_candidate_requires_both_escape_and_return_paths():
    s=server()
    selected,route=asyncio.run(s._select_yield_point('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))
    assert selected.pose.position.y==1.
    assert s._compute_path.await_count==2


def test_point_on_priority_route_rejected_before_nav2_query():
    s=server(); s._yield_points=[goal(1.5,0)]
    with pytest.raises(RuntimeError,match='no reachable'):
        asyncio.run(s._select_yield_point('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))
    s._compute_path.assert_not_called()


def test_escape_path_through_stationary_robot_rejected():
    s=server(); s._compute_path.return_value=[(1.,0.),(0.,0.),(1.,1.)]
    s._compute_path.side_effect=None
    with pytest.raises(RuntimeError,match='no reachable'):
        asyncio.run(s._select_yield_point('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))


def test_return_path_blocked_by_priority_parking_rejected():
    s=server(); s._compute_path.side_effect=[[(1.,0.),(1.,1.)],[(1.,1.),(2.,0.),(-1.,0.)]]
    with pytest.raises(RuntimeError,match='no reachable'):
        asyncio.run(s._select_yield_point('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))


def test_no_points_never_sends_navigation_goal():
    s=server(); s._yield_points=[]
    with pytest.raises(RuntimeError,match='no reachable'):
        asyncio.run(s._select_yield_point('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))


def test_stale_pose_cannot_select_haven():
    s=server(); s._pose_receipt['pinky2']-=3.
    with pytest.raises(RuntimeError,match='fresh tracked pose'):
        asyncio.run(s._select_yield_point('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))


def test_reverse_order_can_avoid_any_evacuation():
    s=server(); s._evacuate=AsyncMock()
    result=asyncio.run(s._preflight([task('pinky1',2,0),task('pinky2',1,2)]))
    assert result is None
    assert s._effective_priority_robot=='pinky2'
    s._evacuate.assert_not_called()


def test_blocked_start_evacuates_then_replans_original_goals():
    s=server()
    async def evacuate(priority,robot,route,original):
        assert priority=='pinky1' and robot=='pinky2'
        assert original.pose.position.x==-1.
        s._poses[robot].pose.pose.position.y=1.
    s._evacuate=AsyncMock(side_effect=evacuate)
    asyncio.run(s._preflight([task('pinky1',2,0),task('pinky2',-1,0)]))
    s._evacuate.assert_awaited_once()
    assert s._compute_path.await_count==4


def evacuation_server(status=GoalStatus.STATUS_SUCCEEDED):
    s=server()
    s._select_yield_point=AsyncMock(return_value=(goal(1,1),[(1.,0.),(1.,1.)]))
    future=NS(done=lambda:True,result=lambda:NS(status=status))
    handle=NS(accepted=True,get_result_async=lambda:future,cancel_goal_async=Mock())
    send=NS(done=lambda:True,result=lambda:handle)
    client=NS(server_is_ready=lambda:True,send_goal_async=Mock(return_value=send))
    s._client=lambda _:client
    s._await_planning=MethodType(MissionActionServer._await_planning,s)
    s._poses['pinky2'].pose.pose.position.y=1.
    return s,handle,client


def test_success_does_not_overwrite_original_goal_or_mark_task_completed():
    s,handle,client=evacuation_server()
    original=goal(-1,0); s._nav_targets['pinky2']=original
    asyncio.run(s._evacuate('pinky1','pinky2',[(0.,0.),(2.,0.)],original))
    assert s._nav_targets['pinky2'] is original
    assert not s._completed_robots and not s._active_nav_goals
    assert client.send_goal_async.call_args.args[0].pose.pose.position.y==1.


def test_success_requires_tracked_arrival():
    s,_,_=evacuation_server(); s._poses['pinky2'].pose.pose.position.y=0.
    s.params['yield_detour_timeout']=.001
    with pytest.raises(RuntimeError,match='has not confirmed'):
        asyncio.run(s._evacuate('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))


def test_cancel_during_selection_never_starts_motion():
    s,_,client=evacuation_server()
    async def select(*args):
        s._mission_goal.is_cancel_requested=True
        return goal(1,1),[(1.,0.),(1.,1.)]
    s._select_yield_point=select
    with pytest.raises(RuntimeError,match='cancelled before'):
        asyncio.run(s._evacuate('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))
    client.send_goal_async.assert_not_called()


def test_unconfirmed_cancel_keeps_handle_and_blocks_new_mission():
    s,handle,_=evacuation_server()
    handle.get_result_async=lambda:NS(done=lambda:False)
    s.params['yield_detour_timeout']=0.
    s._await_planning=AsyncMock(side_effect=TimeoutError('cancel unconfirmed'))
    # Accept is immediate; only the subsequent cancellation confirmation fails.
    s._await_planning.side_effect=[handle,TimeoutError('cancel unconfirmed')]
    with pytest.raises(TimeoutError):
        asyncio.run(s._evacuate('pinky1','pinky2',[(0.,0.),(2.,0.)],goal(-1,0)))
    assert s._active_nav_goals['pinky2'] is handle
    s._mission_reserved=False
    assert MissionActionServer.goal_callback(s,NS(tasks=[]))==GoalResponse.REJECT


def test_points_registration_does_not_enable_robot():
    s=server(); s._mission_active=False
    message=PoseArray(); message.header.frame_id='map'; message.poses=[goal(1,1).pose]
    MissionActionServer._yield_points_callback(s,message)
    assert len(s._yield_points)==1
    assert not s._active_nav_goals
    s._mission_active=True; message.poses=[]
    MissionActionServer._yield_points_callback(s,message)
    assert len(s._yield_points)==1
