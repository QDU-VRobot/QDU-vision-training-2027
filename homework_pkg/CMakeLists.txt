cmake_minimum_required(VERSION 3.10)
project(homework_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 找到所有依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)

# 发布节点
add_executable(ImagePublishNode src/ImagePublishNode.cpp)
ament_target_dependencies(ImagePublishNode
  rclcpp
  sensor_msgs
  cv_bridge
  std_srvs
)
target_link_libraries(ImagePublishNode
  ${OpenCV_LIBRARIES}
)

# 订阅节点
add_executable(ImageSubscribeNode src/ImageSubscribeNode.cpp)
ament_target_dependencies(ImageSubscribeNode
  rclcpp
  sensor_msgs
  cv_bridge
  std_srvs
)
target_link_libraries(ImageSubscribeNode
  ${OpenCV_LIBRARIES}
)

install(TARGETS
  ImagePublishNode
  ImageSubscribeNode
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()