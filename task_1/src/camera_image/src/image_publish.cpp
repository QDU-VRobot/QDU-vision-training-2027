#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

class ImagePublishNode : public rclcpp::Node
{
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;
public:
    explicit ImagePublishNode (const std::string& node_name) : Node(node_name)
    {
        cap_.open(0);
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image", 10);
        timer_ = this->create_wall_timer(1000ms,std::bind(&ImagePublishNode::time_callback,this));
    };

    void time_callback()
    {
        cv::Mat frame;
        cap_ >> frame;

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(),"bgr8", frame).toImageMsg();
        //获取消息头
        msg->header.stamp = this->now();
        msg->header.frame_id = "camera_frame";

        publisher_->publish(*msg);
    };
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImagePublishNode>("image_publisher");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
