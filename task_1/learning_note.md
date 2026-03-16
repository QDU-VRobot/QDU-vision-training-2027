# 学习记录：任务一

## 遇到的问题及解决

### 一、话题名不一致导致收不到图像
**现象**：订阅节点始终收不到图像，日志无输出。  
**原因**：发布节点使用话题 "/image"，而订阅节点写的是 "image"（缺少前导斜杠）。在 ROS2 中，话题名是区分斜杠的，/image 和 image 被视为不同话题。  
**解决**：统一话题名，全部使用带斜杠的绝对名称，如 "/image"。

### 二、处理后的图像未发布
**现象**：Foxglove 中看不到 /image_processed 话题。  
**原因**：在 image_callback 中创建了处理后的消息对象 process_msg，但忘记调用 publish。  
**解决**：补上 processed_publisher_->publish(*process_msg);。

### 三、编译时找不到头文件
**现象**：编译时报错找不到某些头文件。  
**原因**：CMakeLists.txt 中未正确包含依赖项或路径。  
**解决**：使用 compile_commands.json 辅助排查，并确保 find_package 和 target_link_libraries 正确。

## 代码说明
- 第一个 cpp 节点：仅发布原始图像话题 "/image"。
- 第二个 cpp 节点：订阅 "/image" 话题，处理图像后发布到 "/image_processed" 话题，并提供一个服务（无客户端调用），用于演示服务创建。
