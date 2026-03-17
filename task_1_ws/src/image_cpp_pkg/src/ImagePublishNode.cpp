#include <opencv2/opencv.hpp>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "chrono"//时间
#include "sensor_msgs/msg/image.hpp"//ros2中图像消息类型
#include "cv_bridge/cv_bridge.h"//用于在ROS图像消息和OpenCV图像之间转换

class ImagePublishNode : public rclcpp::Node
{
public:
    ImagePublishNode() : Node("image_publish_node")
    {
        //声明参数
        this->declare_parameter("gain", 1.5);
        this->declare_parameter("exposure_time", 25.0);
        //获取参数
        this->get_parameter("gain",gain);
        this->get_parameter("exposure_time",exposure_time);


        //创建发布者
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/image", 10);
        
        //打开摄像头
        camera_.open(0);
        if(!camera_.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "failed");
            return;
        }
        
        //创建定时器
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000), //调用回调的周期
            std::bind(&ImagePublishNode::timer_callback, this));//绑定回调函数
            
        RCLCPP_INFO(this->get_logger(), "success");
    }
    
    ~ImagePublishNode()
    {
        if(camera_.isOpened())
        {
            camera_.release();
        }
        cv::destroyAllWindows();
    }

private:
    void timer_callback()
    {
        cv::Mat frame;
        camera_ >> frame;
        
        if(frame.empty())
        {
            RCLCPP_WARN(this->get_logger(), "failed,frame is empty");
            return;
        }
        
        //显示图像
        cv::imshow("camera", frame);
        cv::waitKey(1);
        
        
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();//将OpenCV图像转换为ROS图像消息

        msg->header.stamp = this->get_clock()->now();//时间戳

        publisher_->publish(*msg);
    }
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;//发布者共享指针
    rclcpp::TimerBase::SharedPtr timer_;//定时器共享指针
    cv::VideoCapture camera_;

    double gain;
    double exposure_time;
};

int main(int argc, char* argv[]) 
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImagePublishNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}