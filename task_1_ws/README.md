一、ImageSubscribeNode


1、共享指针保存数据
// 第一次回调
void data_callback(const sensor_msgs::msg::Image::SharedPtr msg)  // msg指向图像1
{
    // 状态：msg指向图像1（引用计数1）
    //      last_image_ 为空（引用计数0）
    
    last_image_ = msg;
    // 状态：msg指向图像1（引用计数2）
    //      last_image_指向图像1（引用计数2）
    
    print_image_info(msg);
}
// 第二次回调
void data_callback(const sensor_msgs::msg::Image::SharedPtr msg)  // msg指向图像2
{
    // 状态：msg指向图像2（引用计数1）
    //      last_image_指向图像1（引用计数1）
    
    last_image_ = msg;
    // 状态变化：
    // 1. last_image_ 原来指向的图像1（引用计数减1→0，自动释放）
    // 2. last_image_ 现在指向图像2（引用计数加1）
    // 3. msg指向图像2（引用计数2）
    // 4. last_image_指向图像2（引用计数2）
    
    print_image_info(msg);
}


2.try-catch捕获并处理异常
try中通过throw抛出错误码，由catch捕获，try中剩余代码不再执行


3.Trigger服务被选中的根本原因是：保存图像是一个"触发式"操作，不需要输入参数，只需要知道成功与否。
// 保存图像就像按相机快门
// - 按快门：不需要参数（Trigger的空请求）
// - 得到结果：成功/失败，照片文件名（Trigger的响应）

void save_image_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,  // 空请求
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)      // 返回结果
{
    // 就像按快门：不需要任何输入
    (void)request;  // 忽略空请求
    
    // 执行保存
    bool success = save_image();
    
    // 返回结果
    response->success = success;        // 成功了吗？
    response->message = filename;       // 照片文件名
}







二、食用方法

在homework_ws下
source install/setup.bash
ros2 launch image_cpp_pkg launch.py

在foxglove中
sudo apt update
sudo apt installros-humble-foxglove-bridge
ros2 run foxglove_bridge foxglove_bridge
打开另一个终端
cd ~/ROS2/task_1/task_1_ws
colcon build
source install/setup.bash
ros2 launch image_cpp_pkg launch.py


