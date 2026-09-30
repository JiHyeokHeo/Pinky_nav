"""Domain 52 action proxy with per-robot localization interlock."""

import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from pinky_multi_robot.nav2_action_proxy import (
    Nav2ActionProxy, RobotTarget, _parse_args,
)


class LocalizationStates:
    def __init__(self):
        self._lock = threading.Lock()
        self._states = {}

    def update(self, robot, state):
        with self._lock:
            self._states[robot] = (state, time.monotonic())

    def get(self, robot):
        with self._lock:
            state, received = self._states.get(robot, ('WAITING_FOR_GUARD', 0.0))
        return state if time.monotonic() - received < 2.5 else 'WAITING_FOR_GUARD'


class GuardedNav2ActionProxy(Nav2ActionProxy):
    def __init__(self, context, targets, remote_clients, remote_planners, localization):
        self._localization = localization
        super().__init__(context, targets, remote_clients)
        self._remote_planners = remote_planners
        self._plan_goals = {}
        self._plan_servers = [
            ActionServer(
                self, ComputePathToPose, f'/{target.name}/compute_path_to_pose',
                execute_callback=self._make_plan_execute(target),
                goal_callback=lambda _request: GoalResponse.ACCEPT,
                cancel_callback=self._cancel_plan)
            for target in targets
        ]
        self._status_publishers = {
            target.name: self.create_publisher(
                String, f'/{target.name}/localization_status', 10)
            for target in targets
        }
        self.create_timer(1.0, self._publish_status)

    def _publish_status(self):
        for name, publisher in self._status_publishers.items():
            publisher.publish(String(data=self._localization.get(name)))

    async def _execute(self, local_goal, target):
        state = self._localization.get(target.name)
        if state != 'READY':
            self.get_logger().warn(
                f'{target.name}: navigation blocked; localization is {state}.')
            local_goal.abort()
            result = NavigateToPose.Result()
            result.error_msg = f'Localization not ready: {state}'
            return result
        return await super()._execute(local_goal, target)

    def _cancel_plan(self, local_goal):
        remote = self._plan_goals.get(self._key(local_goal))
        if remote is not None:
            remote.cancel_goal_async()
        return CancelResponse.ACCEPT

    def _make_plan_execute(self, target):
        async def execute(local_goal):
            result = ComputePathToPose.Result()
            if self._localization.get(target.name) != 'READY':
                result.error_msg = 'Localization not ready.'
                local_goal.abort()
                return result
            remote_client = self._remote_planners[target.name]
            try:
                while not remote_client.wait_for_server(timeout_sec=0.2):
                    if local_goal.is_cancel_requested:
                        local_goal.canceled()
                        return result
                remote = await remote_client.send_goal_async(local_goal.request)
                if not remote.accepted:
                    result.error_msg = 'Robot planner rejected the path request.'
                    local_goal.abort()
                    return result
                key = self._key(local_goal)
                self._plan_goals[key] = remote
                if local_goal.is_cancel_requested:
                    remote.cancel_goal_async()
                response = await remote.get_result_async()
                result.path = response.result.path
                result.planning_time = response.result.planning_time
                result.error_code = response.result.error_code
                result.error_msg = response.result.error_msg
                if (local_goal.is_cancel_requested or
                        response.status == GoalStatus.STATUS_CANCELED):
                    local_goal.canceled()
                elif response.status == GoalStatus.STATUS_SUCCEEDED:
                    local_goal.succeed()
                else:
                    local_goal.abort()
                return result
            except Exception as error:
                self.get_logger().error(f'{target.name} planning relay failed: {error}')
                result.error_msg = str(error)
                if local_goal.is_cancel_requested:
                    local_goal.canceled()
                else:
                    local_goal.abort()
                return result
            finally:
                self._plan_goals.pop(self._key(local_goal), None)
        return execute


def main(args=None):
    options, ros_args = _parse_args(args)
    targets = (
        RobotTarget('pinky1', options.pinky1_domain, '/pinky1/navigate_to_pose'),
        RobotTarget('pinky2', options.pinky2_domain, '/pinky2/navigate_to_pose'),
    )
    contexts = [Context() for _ in range(3)]
    rclpy.init(args=ros_args, context=contexts[0], domain_id=options.central_domain)
    for context, target in zip(contexts[1:], targets):
        rclpy.init(args=ros_args, context=context, domain_id=target.domain_id)

    localization = LocalizationStates()
    remote_nodes, remote_executors, threads = [], [], []
    central_node = central_executor = None
    try:
        clients = {}
        planners = {}
        for context, target in zip(contexts[1:], targets):
            node = Node(f'{target.name}_guarded_nav2_client', context=context)
            clients[target.name] = ActionClient(node, NavigateToPose, target.robot_action)
            planners[target.name] = ActionClient(
                node, ComputePathToPose,
                target.robot_action.replace('/navigate_to_pose', '/compute_path_to_pose'))
            node.create_subscription(
                String, '/localization_status',
                lambda message, name=target.name: localization.update(name, message.data),
                10)
            executor = MultiThreadedExecutor(num_threads=2, context=context)
            executor.add_node(node)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            remote_nodes.append(node)
            remote_executors.append(executor)
            threads.append(thread)
        central_node = GuardedNav2ActionProxy(
            contexts[0], targets, clients, planners, localization)
        central_executor = MultiThreadedExecutor(num_threads=4, context=contexts[0])
        central_executor.add_node(central_node)
        central_executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if central_executor:
            central_executor.shutdown()
        for executor in remote_executors:
            executor.shutdown()
        if central_node:
            central_node.destroy_node()
        for node in remote_nodes:
            node.destroy_node()
        for context in contexts:
            if context.ok():
                rclpy.shutdown(context=context)
        for thread in threads:
            thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
