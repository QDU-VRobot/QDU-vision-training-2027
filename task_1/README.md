# 任务一：视觉识别功能包

## 功能描述
本功能包包含两个节点：
1. `image_publisher`：发布摄像头原始图像到话题 `/image`。
2. `image_processor`：订阅 `/image` 话题，对图像进行处理（如转换颜色空间、边缘检测等），并将处理后的图像发布到 `/image_processed` 话题；同时提供一个服务 `/process_image`（当前无客户端调用）。

## 环境依赖
- Ubuntu 22.04 / 20.04
- ROS2 Humble / Foxy（根据实际使用版本）
- OpenCV 4.x
- 编译工具：colcon

## 编译方法
在 ROS2 工作空间根目录下执行：
```bash
colcon build --packages-select my_camera_cpp
source install/setup.bash
