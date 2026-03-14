#include <memory>
#include <fstream>
#include <vector>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "std_srvs/srv/trigger.hpp"//触发保存图像

class ImageSubscribeNode : public rclcpp::Node
{
public:
    ImageSubscribeNode() : Node("image_subscribe_node")
    {
        //声明参数
        declare_parameters();
        
        //获取参数初始值
        update_parameters();
        
        //创建订阅者
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image", 10,
            std::bind(&ImageSubscribeNode::data_callback, this, std::placeholders::_1));
        
        //创建处理后图像的发布者
        processed_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/image_processed", 10);
        
        //创建服务
        save_image_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/save_image",
            std::bind(&ImageSubscribeNode::save_image_callback, this, 
                      std::placeholders::_1, std::placeholders::_2));
        
        //创建参数回调
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&ImageSubscribeNode::parameters_callback, this, std::placeholders::_1));
        
        print_parameters();
    }

private:
    // 声明所有参数
    void declare_parameters()
    {
        // 二值化参数
        this->declare_parameter<int>("threshold_value", 128);
        this->declare_parameter<int>("threshold_max_value", 255);
        this->declare_parameter<int>("threshold_type", 0); // 0:二进制, 1:反二进制
        
        // 形态学运算参数
        this->declare_parameter<int>("morph_operation", 0); // 0:腐蚀, 1:膨胀, 2:开运算, 3:闭运算
        this->declare_parameter<int>("morph_iterations", 1);
        this->declare_parameter<int>("morph_kernel_size", 3);
        this->declare_parameter<int>("morph_kernel_shape", 0); // 0:矩形, 1:十字, 2:椭圆
        
        // 轮廓处理参数
        this->declare_parameter<bool>("find_contours", true);
        this->declare_parameter<int>("contour_retrieval_mode", 0); // 0:外部轮廓, 1:所有轮廓
        this->declare_parameter<int>("contour_approximation_mode", 0); // 0:简单, 1:无压缩
        this->declare_parameter<int>("min_contour_area", 100);
        this->declare_parameter<int>("max_contour_area", 100000);
        this->declare_parameter<bool>("draw_contours", true);
        this->declare_parameter<int>("contour_color_b", 0);
        this->declare_parameter<int>("contour_color_g", 255);
        this->declare_parameter<int>("contour_color_r", 0);
        this->declare_parameter<int>("contour_thickness", 2);
        
        // 显示参数
        this->declare_parameter<bool>("show_processed", true);
    }
    
    // 更新参数值
    void update_parameters()
    {
        threshold_value_ = this->get_parameter("threshold_value").as_int();
        threshold_max_value_ = this->get_parameter("threshold_max_value").as_int();
        threshold_type_ = this->get_parameter("threshold_type").as_int();
        
        morph_operation_ = this->get_parameter("morph_operation").as_int();
        morph_iterations_ = this->get_parameter("morph_iterations").as_int();
        morph_kernel_size_ = this->get_parameter("morph_kernel_size").as_int();
        morph_kernel_shape_ = this->get_parameter("morph_kernel_shape").as_int();
        
        find_contours_ = this->get_parameter("find_contours").as_bool();
        contour_retrieval_mode_ = this->get_parameter("contour_retrieval_mode").as_int();
        contour_approximation_mode_ = this->get_parameter("contour_approximation_mode").as_int();
        min_contour_area_ = this->get_parameter("min_contour_area").as_int();
        max_contour_area_ = this->get_parameter("max_contour_area").as_int();
        draw_contours_ = this->get_parameter("draw_contours").as_bool();
        contour_color_b_ = this->get_parameter("contour_color_b").as_int();
        contour_color_g_ = this->get_parameter("contour_color_g").as_int();
        contour_color_r_ = this->get_parameter("contour_color_r").as_int();
        contour_thickness_ = this->get_parameter("contour_thickness").as_int();
        
        show_processed_ = this->get_parameter("show_processed").as_bool();
    }
    
    // 打印当前参数
    void print_parameters()
    {
        RCLCPP_INFO(this->get_logger(), "=== Current Parameters ===");
        RCLCPP_INFO(this->get_logger(), "Threshold: %d/%d, type=%d", 
                   threshold_value_, threshold_max_value_, threshold_type_);
        RCLCPP_INFO(this->get_logger(), "Morph: op=%d, iter=%d, kernel=%d", 
                   morph_operation_, morph_iterations_, morph_kernel_size_);
        RCLCPP_INFO(this->get_logger(), "Contours: enabled=%d, min_area=%d", 
                   find_contours_, min_contour_area_);
    }
    
    // 参数回调函数
    rcl_interfaces::msg::SetParametersResult parameters_callback(
        const std::vector<rclcpp::Parameter>& parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        
        for (const auto& param : parameters)
        {
            RCLCPP_INFO(this->get_logger(), "Parameter '%s' changed", 
                       param.get_name().c_str());
        }
        
        update_parameters();
        print_parameters();
        
        return result;
    }
    
    // 图像处理函数
    cv::Mat process_image(const cv::Mat& input_image)
    {
        if (input_image.empty())
        {
            return cv::Mat();
        }
        
        cv::Mat gray, binary, processed;
        
        // 转换为灰度图
        if (input_image.channels() == 3)
        {
            cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = input_image.clone();
        }
        
        // 1. 二值化
        int thresh_type = (threshold_type_ == 0) ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV;
        cv::threshold(gray, binary, threshold_value_, threshold_max_value_, thresh_type);
        processed = binary.clone();
        
        // 2. 形态学运算
        // 创建结构元素
        int kernel_shape = cv::MORPH_RECT;
        if (morph_kernel_shape_ == 1) kernel_shape = cv::MORPH_CROSS;
        else if (morph_kernel_shape_ == 2) kernel_shape = cv::MORPH_ELLIPSE;
        
        cv::Mat kernel = cv::getStructuringElement(
            kernel_shape, 
            cv::Size(morph_kernel_size_, morph_kernel_size_)
        );
        
        // 应用形态学运算
        switch (morph_operation_)
        {
            case 0: // 腐蚀
                cv::erode(processed, processed, kernel, cv::Point(-1,-1), morph_iterations_);
                break;
            case 1: // 膨胀
                cv::dilate(processed, processed, kernel, cv::Point(-1,-1), morph_iterations_);
                break;
            case 2: // 开运算
                cv::morphologyEx(processed, processed, cv::MORPH_OPEN, kernel, 
                                cv::Point(-1,-1), morph_iterations_);
                break;
            case 3: // 闭运算
                cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel, 
                                cv::Point(-1,-1), morph_iterations_);
                break;
        }
        
        // 3. 轮廓处理
        if (find_contours_ && !processed.empty())
        {
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            
            // 检索模式
            int retrieval_mode = cv::RETR_EXTERNAL;
            if (contour_retrieval_mode_ == 1) retrieval_mode = cv::RETR_LIST;
            else if (contour_retrieval_mode_ == 2) retrieval_mode = cv::RETR_TREE;
            
            // 近似模式
            int approx_mode = cv::CHAIN_APPROX_SIMPLE;
            if (contour_approximation_mode_ == 1) approx_mode = cv::CHAIN_APPROX_NONE;
            
            cv::findContours(processed, contours, hierarchy, retrieval_mode, approx_mode);
            
            if (draw_contours_)
            {
                // 创建彩色图像用于绘制轮廓
                cv::Mat contour_img;
                if (input_image.channels() == 3)
                {
                    contour_img = input_image.clone();
                }
                else
                {
                    cv::cvtColor(input_image, contour_img, cv::COLOR_GRAY2BGR);
                }
                
                cv::Scalar color(contour_color_b_, contour_color_g_, contour_color_r_);
                int valid_contours = 0;
                
                for (size_t i = 0; i < contours.size(); i++)
                {
                    double area = cv::contourArea(contours[i]);
                    if (area >= min_contour_area_ && area <= max_contour_area_)
                    {
                        cv::drawContours(contour_img, contours, i, color, contour_thickness_);
                        valid_contours++;
                    }
                }
                
                RCLCPP_DEBUG(this->get_logger(), "Found %zu contours, drew %d", 
                           contours.size(), valid_contours);
                return contour_img;
            }
            
            RCLCPP_DEBUG(this->get_logger(), "Found %zu contours", contours.size());
        }
        
        return processed;
    }
    
    void data_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 保存最后一帧图像
        last_image_ = msg;
        
        
        try
        {
            // 转换为OpenCV图像
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            
            // 处理图像
            cv::Mat processed = process_image(cv_ptr->image);
            
            if (!processed.empty())
            {
                // 发布处理后的图像
                std::string encoding = (processed.channels() == 3) ? "bgr8" : "mono8";
                auto processed_msg = cv_bridge::CvImage(
                    std_msgs::msg::Header(), encoding, processed).toImageMsg();
                processed_msg->header.stamp = this->get_clock()->now();
                processed_msg->header.frame_id = "camera_frame_processed";
                processed_publisher_->publish(*processed_msg);
                
                // 显示处理后的图像
                if (show_processed_)
                {
                    cv::imshow("Processed Image", processed);
                    cv::waitKey(1);
                }
            }
            
            // 打印原始图像信息
            print_image(msg);
        }
        catch (const cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
        }
    }
    
    //服务回调函数：保存最后一帧图像
    void save_image_callback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,//服务请求（输入）
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)//服务响应（输出）
    {
        (void)request;//忽略参数
        
        if (!last_image_)
        {
            response->success = false;
            response->message = "No image received yet";
            RCLCPP_WARN(this->get_logger(), "Save image failed: no image received yet");
            return;
        }
        
        try
        {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(last_image_, last_image_->encoding);
            cv::Mat image = cv_ptr->image;
            cv::Mat processed = process_image(image);
            
            //命名为时间戳
            auto now = this->now();
            std::string filename = "image_" + std::to_string(now.seconds()) + ".png";
            std::string proc_filename = "processed_" + std::to_string(now.seconds()) + ".png";
            
            bool save_orig = cv::imwrite(filename, image);//保存原始图像
            bool save_proc = cv::imwrite(proc_filename, processed);//保存处理后图像
            
            if (save_orig && save_proc)
            {
                response->success = true;
                response->message = "Saved: " + filename + " and " + proc_filename;
                RCLCPP_INFO(this->get_logger(), "Images saved successfully");
            }
            else
            {
                response->success = false;
                response->message = "Failed to save images";
                RCLCPP_ERROR(this->get_logger(), "Failed to save images");
            }
        }
        catch (const std::exception& e)//捕获所有标准异常
        {
            response->success = false;
            response->message = std::string("Error: ") + e.what();
            RCLCPP_ERROR(this->get_logger(), "Save failed: %s", e.what());
        }
    }
    
    void print_image(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "========== Received Image ==========");
        RCLCPP_INFO(this->get_logger(), "--- Message Header ---");
        RCLCPP_INFO(this->get_logger(), "Timestamp: %d.%u", 
                   msg->header.stamp.sec, msg->header.stamp.nanosec);
        RCLCPP_INFO(this->get_logger(), "Frame ID: '%s'", 
                   msg->header.frame_id.c_str());
        RCLCPP_INFO(this->get_logger(), "--- Image Properties ---");
        RCLCPP_INFO(this->get_logger(), "Size: %d x %d", msg->width, msg->height);
        RCLCPP_INFO(this->get_logger(), "Encoding: %s", msg->encoding.c_str());
        RCLCPP_INFO(this->get_logger(), "Data Size: %zu bytes", msg->data.size());
        RCLCPP_INFO(this->get_logger(), "=====================================\n");
    }
    
    // 成员变量
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_publisher_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_image_service_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    
    sensor_msgs::msg::Image::SharedPtr last_image_;
    
    // 参数变量
    int threshold_value_;
    int threshold_max_value_;
    int threshold_type_;
    
    int morph_operation_;
    int morph_iterations_;
    int morph_kernel_size_;
    int morph_kernel_shape_;
    
    bool find_contours_;
    int contour_retrieval_mode_;
    int contour_approximation_mode_;
    int min_contour_area_;
    int max_contour_area_;
    bool draw_contours_;
    int contour_color_b_;
    int contour_color_g_;
    int contour_color_r_;
    int contour_thickness_;
    
    bool show_processed_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc,argv);
    auto node = std::make_shared<ImageSubscribeNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}