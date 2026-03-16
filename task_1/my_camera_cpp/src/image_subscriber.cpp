#include "sensor_msgs/msg/image.hpp"
#include <opencv2/core/mat.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <functional>
#include <chrono>
#include <string>
#include <fstream>

class ImageSubscribeNode:public rclcpp::Node
{
    public:
    ImageSubscribeNode(): Node("image_subscriber")
    {
        this->declare_parameter<int>("threshold",128);
        this->declare_parameter<std::string>("leixing","open");
        this->declare_parameter<int>("cishu",1);
        subscription_=this->create_subscription<sensor_msgs::msg::Image>("image", 10, std::bind(&ImageSubscribeNode::image_callback,this,std::placeholders::_1));
        service_ =this->create_service<std_srvs::srv::Trigger>("/save_image",std::bind(&ImageSubscribeNode::save_callback, this,std::placeholders::_1, std::placeholders::_2));
        processed_publisher_=this->create_publisher<sensor_msgs::msg::Image>("/image_processed", 10);
        RCLCPP_INFO(this->get_logger(),"ImageSubscribeNode started.");
    };

    private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(),"Received image: sec=%d, nanosec=%d, frame_id=%s, height=%d, width=%d, encoding=%s",msg->header.stamp.sec,msg->header.stamp.nanosec,
        msg->header.frame_id.c_str(),msg->height,msg->width,msg->encoding.c_str());

        //图像可能转化失败，用try兜住。
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr =cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        //克隆一份用来服务
        latest_frame_=cv_ptr->image.clone();

        //图像处理函数
        cv::Mat processed=process_image(cv_ptr->image);
        //将处理后的图像发布出去，发布者。发出去的时候还得转成ros信息格式。mono8无符号整形灰度图。
        auto process_msg=cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", processed).toImageMsg();
        processed_publisher_->publish(*process_msg);
    };
    //发布以时间戳命名的从发布者接受由订阅者转化的图像
    void save_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;//无客户端请求
        if(latest_frame_.empty())
        {
            response->success=false;
            response->message="No image";
            RCLCPP_WARN(this->get_logger(),"no image");
            return;
        }
        //生成时间戳文件名
        auto now=std::chrono::system_clock::now();//获取当前时间
        auto now_c=std::chrono::system_clock::to_time_t(now);//把时间转化成秒
        auto now_ms=std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch())%1000;//通过纪元时间精确到毫秒，防止一秒内多次保存导致文件名相同

        std::stringstream ss;
        ss << "image_" << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S")
           << '_' << now_ms.count() << ".png";
        std::string filename = ss.str();

        //保存图片
        if (cv::imwrite(filename, latest_frame_))
        {
            response->success = true;
            response->message = "Saved as " + filename;
            RCLCPP_INFO(this->get_logger(), "Image saved: %s", filename.c_str());
        }
        else
        {
            response->success = false;
            response->message = "Failed to save image.";
            RCLCPP_ERROR(this->get_logger(), "Failed to save image.");
        }


    };
    cv::Mat process_image(const cv::Mat& input)
    {
        //获取参数
        int threshold_val = this->get_parameter("threshold").as_int();
        std::string leixing = this->get_parameter("leixing").as_string();
        int cishu = this->get_parameter("cishu").as_int();

        // 转换为灰度图
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

        // 二值化
        cv::Mat binary;
        cv::threshold(gray, binary, threshold_val, 255, cv::THRESH_BINARY);

        // 形态学操作
        cv::Mat morph_result;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        if (leixing == "open")
        {
            cv::morphologyEx(binary, morph_result, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), cishu);
        }
        else if (leixing == "close")
        {
            cv::morphologyEx(binary, morph_result, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), cishu);
        }
        else if (leixing == "erode")
        {
            cv::erode(binary, morph_result, kernel, cv::Point(-1,-1), cishu);
        }
        else if (leixing == "dilate")
        {
            cv::dilate(binary, morph_result, kernel, cv::Point(-1,-1), cishu);
        }
        else
        {
            morph_result = binary; // 默认不做
        }
        return morph_result;
    };
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr service_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_publisher_;
    cv::Mat latest_frame_;
};

int main(int argc,char * argv[])
{
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<ImageSubscribeNode>());
    rclcpp::shutdown();
    return 0;
}