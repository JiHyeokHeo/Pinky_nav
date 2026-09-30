# A task for one robot. robot_id is the Domain 50 alias: pinky1 or pinky2.
string robot_id

uint8 PARK=0
uint8 PATROL=1
uint8 STOP=2
uint8 task_type

# PARK uses parking_goal. PATROL uses patrol_waypoints in order.
geometry_msgs/PoseStamped parking_goal
geometry_msgs/PoseStamped[] patrol_waypoints

# PATROL: 0 means patrol until the mission goal is cancelled.
uint32 patrol_laps
float32 pause_seconds
