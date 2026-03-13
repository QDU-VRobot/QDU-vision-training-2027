# QDU-vision-training-2026

27赛季算法组培训作业提交仓库

## 提交要求

项目必须使用 git 进行管理，包含必要的提交记录，commit message需包含有效信息，尽量不要出现 `update`、`111`等信息。

第一次提交请创建分支，名为姓名拼音，此后的作业提交至该分支中。

目录结构应如下：

```bash
.
├── task_1 # 任务一工作区
├── task_2 # 任务二工作区
├── task_3 # 任务三工作区
......
```

需包含已实现任务点的完整源码、README文档、必要的学习记录文档。

不可抄袭、滥用 AI，代码补全人之常情，"给出完整代码"不可接受。

## 当前已布置任务

### 任务一

1. 编写两个ROS2节点，节点一 `ImagePublishNode` 通过OpenCV打开笔记本摄像头，将图像发布至话题 `/image`；节点二 `ImageSubscribeNode` 订阅话题 `/image` ，通过 ROS2 日志打印图像信息数据，包括消息头、消息所属坐标系、图像数据属性等等。
2. 在 `ImageSubscribeNode` 中添加服务 `/save_image` ，触发时将最近接收的一帧图像保存到当前包下的 `images/` 目录（如不存在需创建），文件名使用该图像消息的 `header.stamp` 作为时间戳，格式为 `<sec>_<nanosec>.png`（例如 `1710326400_123456789.png`）。
3. 添加 launch 文件，启动这两个节点，并通过 `yaml` 文件加载节点参数 `gain`、`exposure_time`等等。
4. 在 `ImageSubscribeNode` 中对接收的图像进行处理，包括二值化、轮廓处理、各类形态学运算，二值化阈值、轮廓处理方法、形态学运算类型/次数作为 ROS2 可配置参数，将处理后的图像发布为 `/image_processed`。
5. 下载并配置 Foxglove，启动 `foxglove_bridge`，在 Foxglove 中调试参数、显示图像。
