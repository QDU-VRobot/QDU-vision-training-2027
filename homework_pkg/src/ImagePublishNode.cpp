#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>

class ImagePublishNode : public rclcpp::Node 
{
public:
    ImagePublishNode() : Node("image_publish_node")
    {
        this->declare_parameter("camera_id", 0);
        this->declare_parameter("frame_width", 640);
        this->declare_parameter("frame_height", 480);

        int camera_id = this->get_parameter("camera_id").as_int();
        int frame_width = this->get_parameter("frame_width").as_int();
        int frame_height = this->get_parameter("frame_height").as_int();

        cap_.open(camera_id);
        if(!cap_.isOpened()){
            RCLCPP_ERROR(this->get_logger(), "摄像头打开失败");
            return;
        }
        
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
        
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&ImagePublishNode::timer_callback, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "图像发布节点启动成功");
    }

private:
    void timer_callback()
    {
        cv::Mat frame;
        cap_ >> frame;
        if(frame.empty()) return;
        
        auto msg = cv_bridge::CvImage(
            std_msgs::msg::Header(),
            "bgr8",
            frame
        ).toImageMsg();
        
        publisher_->publish(*msg);
    }
    
    cv::VideoCapture cap_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImagePublishNode>());
    rclcpp::shutdown();
    return 0;
}