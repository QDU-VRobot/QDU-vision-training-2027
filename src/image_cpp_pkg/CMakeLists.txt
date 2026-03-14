cmake_minimum_required(VERSION 3.8)
project(image_cpp_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(ImagePublishNode src/ImagePublishNode.cpp)
add_executable(ImageSubscribeNode src/ImageSubscribeNode.cpp)

ament_target_dependencies(ImagePublishNode rclcpp sensor_msgs cv_bridge OpenCV std_msgs)
ament_target_dependencies(ImageSubscribeNode rclcpp sensor_msgs cv_bridge OpenCV std_msgs std_srvs)



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

install(TARGETS 
ImagePublishNode ImageSubscribeNode
DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
DESTINATION share/${PROJECT_NAME}
)


install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
