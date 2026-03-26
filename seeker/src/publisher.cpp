#include <bits/stdc++.h>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.hpp>  
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
class Publisher :public rclcpp::Node
{
  public :
  void recall()
  {
    cv::Mat picture;
    camera.open(0);
    camera>>picture;
    auto msg=cv_bridge::CvImage(std_msgs::msg::Header(),"bgr8",picture).toImageMsg();
    msg->header.stamp=this->now();
    publi->publish(*msg);
    RCLCPP_INFO(this->get_logger(),"Is_Opened");

  }
  Publisher(std::string name):Node(name){
    this->declare_parameter<double>("gain",0.0);
    this->declare_parameter<int>("exposure_time",0);
    publi = this->create_publisher<sensor_msgs::msg::Image>("/image",10);
    timer = this->create_wall_timer(std::chrono::milliseconds(500),std::bind(&Publisher::recall,this));
  
  }
  private :
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publi;
  rclcpp::TimerBase::SharedPtr timer;
  cv::VideoCapture camera;
  

};
int main(int argc, char ** argv)
{
  rclcpp::init(argc,argv);
  auto ImagePublishNode = std::make_shared<Publisher>("Node_01");
  rclcpp::spin(ImagePublishNode);
  rclcpp::shutdown();
  return 0;

}
