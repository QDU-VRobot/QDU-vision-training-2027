#include "std_msgs/msg/header.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <rclcpp/executors.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

class ImagePublishNode:public rclcpp:: Node
{
    public:
    ImagePublishNode():Node ("image_publisher")
    {   //发布者
        publisher_ =this->create_publisher<sensor_msgs::msg::Image>("/image",10);
        //定时回调
        timer_ =this->create_wall_timer(std::chrono::milliseconds(100),std::bind(&ImagePublishNode::timer_callback,this));
        //打开摄像头
        cap_.open(0);
        if (!cap_.isOpened()) {           
            RCLCPP_ERROR(this->get_logger(), "can not open camera");
        }else{
            RCLCPP_INFO(this->get_logger(), "camera ok");
        }

    }

    ~ImagePublishNode()
    {
        if (cap_.isOpened())
        cap_.release();
    }

    private:
    void timer_callback()
    {
        cv::Mat frame;
        cap_>>frame;
        if(frame.empty())
        {
            RCLCPP_ERROR(this->get_logger(),"picture empty");
            return;
        }
        
        auto msg=cv_bridge::CvImage(std_msgs::msg::Header(),"bgr8",frame).toImageMsg();
        publisher_ ->publish(*msg);
    };
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
};


int main(int argc,char *argv[])
{
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<ImagePublishNode>());
    rclcpp::shutdown();
    return 0;
}