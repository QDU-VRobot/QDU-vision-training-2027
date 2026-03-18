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
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# === 修改：强制使用系统的 OpenCV ===
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

# === 新增：添加链接器标志，优先使用系统库 ===
add_link_options(-Wl,-rpath,/usr/lib/x86_64-linux-gnu -Wl,--disable-new-dtags)

add_executable(ImagePublishNode src/ImagePublishNode.cpp)
add_executable(ImageSubscribeNode src/ImageSubscribeNode.cpp)

ament_target_dependencies(ImagePublishNode rclcpp sensor_msgs cv_bridge std_msgs)
ament_target_dependencies(ImageSubscribeNode rclcpp sensor_msgs cv_bridge std_msgs std_srvs)

target_link_libraries(ImagePublishNode ${OpenCV_LIBRARIES})
target_link_libraries(ImageSubscribeNode ${OpenCV_LIBRARIES})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS 
  ImagePublishNode 
  ImageSubscribeNode
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()