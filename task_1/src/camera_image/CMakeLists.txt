cmake_minimum_required(VERSION 3.8)
project(camera_image)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(image_publish src/image_publish.cpp)
ament_target_dependencies(image_publish rclcpp OpenCV sensor_msgs cv_bridge)
add_executable(image_subscribe src/image_subscribe.cpp)
ament_target_dependencies(image_subscribe rclcpp OpenCV sensor_msgs cv_bridge std_srvs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS image_publish image_subscribe 
DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config
DESTINATION share/${PROJECT_NAME}
)

ament_package()
