cmake_minimum_required(VERSION 3.8)
project(pinky_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(MSG_FILES "msg/RobotTask.msg")
set(SRV_FILES
  "srv/SetLed.srv"
  "srv/Emotion.srv"
  "srv/SetBrightness.srv"
  "srv/SetLamp.srv")
set(ACTION_FILES "action/ExecuteMultiRobotMission.action")

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES} ${SRV_FILES} ${ACTION_FILES}
  DEPENDENCIES builtin_interfaces std_msgs action_msgs geometry_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
