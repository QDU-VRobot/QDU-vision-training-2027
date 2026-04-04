#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>

class ImageSubscriberNode : public rclcpp::Node
{
public:
    ImageSubscriberNode() : Node("image_subscriber_node")
    {
        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image",
            10,
            std::bind(&ImageSubscriberNode::image_callback, this, std::placeholders::_1)
        );
        
        service_ = this->create_service<std_srvs::srv::Trigger>(
            "save_image",
            std::bind(&ImageSubscriberNode::toggle_callback, this,
                      std::placeholders::_1, std::placeholders::_2)
        );
        
        RCLCPP_INFO(this->get_logger(), "图像订阅节点启动成功");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        if(enable_process_){
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
            cv::Canny(frame, frame, 20, 150);
        }
        cv::imshow("Camera View", frame);
        cv::waitKey(1);
    }
    
    void toggle_callback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        enable_process_ = !enable_process_;
        response->success = true;
        response->message = enable_process_ ? "开启图像处理" : "关闭图像处理";
        RCLCPP_INFO(this->get_logger(), response->message.c_str());
    }
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr service_; 
    bool enable_process_ = false;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageSubscriberNode>());
    rclcpp::shutdown();
    return 0;
}