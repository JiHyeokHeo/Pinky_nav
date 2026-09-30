"""Relay Nav2 NavigateToPose actions between isolated ROS domains.

``domain_bridge`` only bridges topics. This process owns action servers on the
central domain and action clients on each robot domain, relaying the goal,
cancellation, feedback, and result parts of a ROS action.
"""
import argparse
import threading
from dataclasses import dataclass

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


@dataclass(frozen=True)
class RobotTarget:
    name: str
    domain_id: int
    central_action: str
    robot_action: str = '/navigate_to_pose'


@dataclass
class RelaySession:
    remote_goal: object = None
    cancel_sent: bool = False


class Nav2ActionProxy(Node):
    """Expose one central action server for each robot Nav2 action server."""

    def __init__(self, context, targets, remote_clients):
        super().__init__('nav2_action_proxy', context=context)
        self._lock = threading.RLock()
        self._sessions = {}
        self._servers = []
        self._remote_clients = remote_clients
        for target in targets:
            self._servers.append(ActionServer(
                self, NavigateToPose, target.central_action,
                # ActionServer detects async callbacks with inspect. A normal
                # lambda returning a coroutine is not detected or awaited.
                execute_callback=self._make_execute_callback(target),
                goal_callback=lambda _request: GoalResponse.ACCEPT,
                cancel_callback=self._cancel))
        self.get_logger().info('Relaying ' + ', '.join(t.central_action for t in targets))

    @staticmethod
    def _key(goal_handle):
        return bytes(goal_handle.goal_id.uuid)

    def _make_execute_callback(self, target):
        async def execute(goal_handle):
            return await self._execute(goal_handle, target)
        return execute

    def _cancel(self, goal_handle):
        with self._lock:
            session = self._sessions.get(self._key(goal_handle))
            if session and session.remote_goal is not None and not session.cancel_sent:
                session.cancel_sent = True
                session.remote_goal.cancel_goal_async()
        return CancelResponse.ACCEPT

    @staticmethod
    def _forward_feedback(local_goal, feedback_message):
        if not local_goal.is_cancel_requested:
            local_goal.publish_feedback(feedback_message.feedback)

    async def _execute(self, local_goal, target):
        key = self._key(local_goal)
        client = self._remote_clients[target.name]
        session = RelaySession()
        with self._lock:
            self._sessions[key] = session
        try:
            warned_unavailable = False
            while not client.wait_for_server(timeout_sec=0.25):
                if local_goal.is_cancel_requested:
                    local_goal.canceled()
                    return NavigateToPose.Result()
                if not warned_unavailable:
                    self.get_logger().warn(
                        f'Waiting for {target.name} Nav2 action server on '
                        f'Domain {target.domain_id}: {target.robot_action}')
                    warned_unavailable = True
            remote_goal = await client.send_goal_async(
                local_goal.request,
                feedback_callback=lambda msg: self._forward_feedback(local_goal, msg))
            with self._lock:
                session.remote_goal = remote_goal
                if local_goal.is_cancel_requested and not session.cancel_sent:
                    session.cancel_sent = True
                    remote_goal.cancel_goal_async()
            if not remote_goal.accepted:
                self.get_logger().warn(f'{target.name} rejected a navigation goal.')
                local_goal.abort()
                return NavigateToPose.Result()
            remote_response = await remote_goal.get_result_async()
            # Never return the GetResult service response itself: an ActionServer
            # execute callback must return only NavigateToPose.Result.  Copying
            # also keeps Python message objects from the robot Context separate
            # from the central Context.
            remote_result = remote_response.result
            local_result = NavigateToPose.Result()
            local_result.error_code = remote_result.error_code
            local_result.error_msg = remote_result.error_msg
            if (local_goal.is_cancel_requested or
                    remote_response.status == GoalStatus.STATUS_CANCELED):
                local_goal.canceled()
            elif remote_response.status == GoalStatus.STATUS_SUCCEEDED:
                local_goal.succeed()
            else:
                local_goal.abort()
            return local_result
        except Exception as error:
            self.get_logger().error(f'{target.name} action relay failed: {error}')
            if local_goal.is_cancel_requested:
                local_goal.canceled()
            else:
                local_goal.abort()
            return NavigateToPose.Result()
        finally:
            with self._lock:
                self._sessions.pop(key, None)


def _parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--central-domain', type=int, default=52)
    parser.add_argument('--pinky1-domain', type=int, default=20)
    parser.add_argument('--pinky2-domain', type=int, default=22)
    return parser.parse_known_args(args)


def main(args=None):
    options, ros_args = _parse_args(args)
    targets = (
        RobotTarget('pinky1', options.pinky1_domain, '/pinky1/navigate_to_pose'),
        RobotTarget('pinky2', options.pinky2_domain, '/pinky2/navigate_to_pose'),
    )
    contexts = [Context() for _ in range(1 + len(targets))]
    rclpy.init(args=ros_args, context=contexts[0], domain_id=options.central_domain)
    for context, target in zip(contexts[1:], targets):
        rclpy.init(args=ros_args, context=context, domain_id=target.domain_id)

    remote_nodes, remote_executors, threads = [], [], []
    central_node = central_executor = None
    try:
        clients = {}
        for context, target in zip(contexts[1:], targets):
            node = Node(f'{target.name}_nav2_action_client', context=context)
            clients[target.name] = ActionClient(node, NavigateToPose, target.robot_action)
            executor = MultiThreadedExecutor(num_threads=2, context=context)
            executor.add_node(node)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            remote_nodes.append(node)
            remote_executors.append(executor)
            threads.append(thread)
        central_node = Nav2ActionProxy(contexts[0], targets, clients)
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
