# 视觉任务一：ROS2 图像处理节点

## 功能
- `image_publisher`：发布摄像头图像到 `/image`
- `image_subscriber`：订阅 `/image`，打印信息，提供 `/save_image` 服务，进行HSV颜色分割、形态学处理、轮廓检测，并发布 `/image_processed`

## 编译
```bash
cd ~/ros2_study/vision_task/task1_ws
colcon build --packages-select camera_image
source install/setup.bash