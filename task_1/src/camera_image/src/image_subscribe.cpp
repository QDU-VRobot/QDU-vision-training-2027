#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

class ImageSubscribeNode : public rclcpp::Node
{
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publish_;

    std::mutex mutex_;//互斥锁对象，保护共享数据，防止多线程同时访问导致数据混乱
    cv::Mat latest_image_;
    std_msgs::msg::Header latest_header_;
public:
    explicit ImageSubscribeNode(const std::string& node_name) : Node(node_name)
    {
        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>("image", 10, std::bind(&ImageSubscribeNode::image_callback, 
            this, std::placeholders::_1));
        save_service_ = this->create_service<std_srvs::srv::Trigger>("save_image",
            std::bind(&ImageSubscribeNode::save_image_callback, this, std::placeholders::_1, std::placeholders::_2));
        image_publish_ = this->create_publisher<sensor_msgs::msg::Image>("image_processed", 10);

        //声明参数及其初始值
        this->declare_parameter("morph_operator", 2);      
        this->declare_parameter("morph_iterations", 1);
        this->declare_parameter("enable_contour", true);
        this->declare_parameter("h_min", 35);
        this->declare_parameter("h_max", 77);
        this->declare_parameter("s_min", 50);
        this->declare_parameter("s_max", 255);
        this->declare_parameter("v_min", 50);
        this->declare_parameter("v_max", 255);
    };

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        
        //打印信息
        RCLCPP_INFO(this->get_logger(), "收到图像：");
        RCLCPP_INFO(this->get_logger(), "  宽度: %d", msg->width);
        RCLCPP_INFO(this->get_logger(), "  高度: %d", msg->height);
        RCLCPP_INFO(this->get_logger(), "  编码: %s", msg->encoding.c_str());
        RCLCPP_INFO(this->get_logger(), "  步长: %d", msg->step);
        RCLCPP_INFO(this->get_logger(), "  帧ID: %s", msg->header.frame_id.c_str());
        RCLCPP_INFO(this->get_logger(), "  时间戳: %d.%d", msg->header.stamp.sec, msg->header.stamp.nanosec);

        //保存为opencv
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);

        //加锁保护共享数据,(服务通讯多线程时使用)
        //{}限定了一个作用域
        {
            std::lock_guard<std::mutex> lock(mutex_);//自动加锁，出作用域自动解锁
            latest_image_ = cv_ptr->image.clone();//克隆图像，保存副本
            latest_header_ = msg->header;//保存信息头
        }

        //图像处理
        //获取参数（每次回调读取最新值，实现动态调整）
        int h_min = this->get_parameter("h_min").as_int();
        int h_max = this->get_parameter("h_max").as_int();
        int s_min = this->get_parameter("s_min").as_int();
        int s_max = this->get_parameter("s_max").as_int();
        int v_min = this->get_parameter("v_min").as_int();
        int v_max = this->get_parameter("v_max").as_int();
        int morph_op = this->get_parameter("morph_operator").as_int();
        int morph_iter = this->get_parameter("morph_iterations").as_int();
        bool enable_contour = this->get_parameter("enable_contour").as_bool();

        //HSV
        cv::Mat hsv, binary_image;
        cv::cvtColor(cv_ptr->image, hsv, cv::COLOR_BGR2HSV);
        cv::Scalar lower(h_min, s_min, v_min);
        cv::Scalar upper(h_max, s_max, v_max);
        cv::inRange(hsv, lower, upper, binary_image);

        //形态学处理
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat morph;
        switch (morph_op) 
        {
            case 0: // 腐蚀
                cv::erode(binary_image, morph, kernel, cv::Point(-1,-1), morph_iter);
                break;
            case 1: // 膨胀
                cv::dilate(binary_image, morph, kernel, cv::Point(-1,-1), morph_iter);
                break;
            case 2: // 开运算
                cv::morphologyEx(binary_image, morph, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), morph_iter);
                break;
            case 3: // 闭运算
                cv::morphologyEx(binary_image, morph, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), morph_iter);
                break;
            default:
                morph = binary_image;
        };
        //轮廓处理
        cv::Mat output_img;
        std::string encoding;

        if (enable_contour) {
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            output_img = cv_ptr->image.clone();//绘制图像
            cv::drawContours(output_img, contours, -1, cv::Scalar(0, 255, 0), 2);
            encoding = "bgr8";
        } else {
            output_img = morph;//直接输出二值图
            encoding = "mono8";
}

        auto processed_msg = cv_bridge::CvImage(msg->header, encoding, output_img).toImageMsg();
        image_publish_->publish(*processed_msg);
    };

    void save_image_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                   std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;

        //局部变量
        cv::Mat img;
        std_msgs::msg::Header header;
        {
            std::lock_guard<std::mutex> lock(mutex_);//加锁，保护读取，确保读取时不会被修改
            //拷贝一份
            img  = latest_image_.clone();
            header = latest_header_;
        }

        //保存图像
        std::string filename = std::to_string(header.stamp.sec) + "_" +
                             std::to_string(header.stamp.nanosec) + ".png";
        //必须填充response
        if (cv::imwrite(filename, img)) {
            response->success = true;
            response->message = "保存为 " + filename;
            RCLCPP_INFO(this->get_logger(), "已保存: %s", filename.c_str());
        } else {
            response->success = false;
            response->message = "保存失败";
        }
    };

};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImageSubscribeNode>("image_subscriber");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
