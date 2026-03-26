#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/empty.hpp>
cv::Mat mid_save;
cv::Mat pro_image;
long long stamp1,nstamp;
class Subscriber:public rclcpp::Node
{
    private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr ser;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed;
    rclcpp::TimerBase::SharedPtr timer2;
    int thresh;
    int max_thr;
    int type_thr;
    int shape;
    cv::Size ksize;
    int morph_op;
    int ksize_w;
    int ksize_h;
    int times_morph;
    int type_morph;
    int mod_contour;
    int method_contour;
    public:
    
    void recall_proc();
    void recall(sensor_msgs::msg::Image::SharedPtr A);
    void recall_ser(const std::shared_ptr<std_srvs::srv::Empty::Request> a,const std::shared_ptr<std_srvs::srv::Empty::Response> b);
    Subscriber(std::string name):Node(name)
    {
        this->declare_parameter<int>("thresh",120);
        this->declare_parameter<int>("max_thr",255);
        this->declare_parameter<int>("type_thr",0);
        this->declare_parameter<int>("shape",0);
        // this->declare_parameter<cv::Size>("ksize",cv::Size(3,3));
        this->declare_parameter<int>("ksize_w",3);
        this->declare_parameter<int>("ksize_h",3);
        this->declare_parameter<int>("morph_op",0);
        this->declare_parameter<int>("times_morph",1);
        this->declare_parameter<int>("type_morph",1);
        this->declare_parameter<int>("mod_contour",1);
        this->declare_parameter<int>("method_contour",1);
        this->declare_parameter<double>("gain",0.0);
        this->declare_parameter<int>("exposure_time",0);
        thresh=this->get_parameter("thresh").as_int();
        max_thr=this->get_parameter("max_thr").as_int();
        type_thr=this->get_parameter("type_thr").as_int();
        shape=this->get_parameter("shape").as_int();
        ksize_w=this->get_parameter("ksize_w").as_int();
        ksize_h=this->get_parameter("ksize_h").as_int();
        morph_op=this->get_parameter("morph_op").as_int();
        times_morph=this->get_parameter("times_morph").as_int();
        type_morph=this->get_parameter("type_morph").as_int();
        mod_contour=this->get_parameter("mod_contour").as_int();
        method_contour=this->get_parameter("method_contour").as_int();
        timer2=this->create_wall_timer(std::chrono::milliseconds(500),std::bind(&Subscriber::recall_proc,this));
        processed=this->create_publisher<sensor_msgs::msg::Image>("/image_processed",10);
        sub=rclcpp::create_subscription<sensor_msgs::msg::Image>(*this,"/image",10,std::bind(&Subscriber::recall,this,std::placeholders::_1));
        ser=this->create_service<std_srvs::srv::Empty>("/save_image",std::bind(&Subscriber::recall_ser,this,std::placeholders::_1,std::placeholders::_2));
    }

};
void Subscriber::recall_proc()
{
    if(mid_save.empty()) {
        RCLCPP_INFO(this->get_logger(),"it is empty");
        return;  
    }
    // cv::Mat mid_save_cpy=mid_save.clone();
    // cv::Mat gray,pro_image_2;
    // cv::cvtColor(mid_save_cpy,gray,cv::COLOR_BGR2GRAY);
    // cv::threshold(gray,pro_image,thresh,max_thr,type_thr);
    // cv::Mat ker=cv::getStructuringElement(shape,cv::Size(ksize_w,ksize_h));
    // cv::Mat mid_pro;
    // pro_image_2=pro_image.clone();
    // cv::morphologyEx(pro_image_2,mid_pro,morph_op,ker,cv::Point(-1,-1),times_morph);
    // std::vector<std::vector<cv::Point>> mmmm;
    // cv::cvtColor(mid_pro,mid_pro,cv::COLOR_BGR2GRAY);
    // cv::findContours(mid_pro,mmmm,mod_contour,method_contour);
    // cv::drawContours(mid_save_cpy,mmmm,-1,cv::Scalar(0,0,255),2);
    //图像处理这里有个段错误，一直没调好，时间截止了就先不改了
    auto msg=cv_bridge::CvImage(std_msgs::msg::Header(),"bgr8",mid_save).toImageMsg();
    msg->header.stamp=this->now();
    processed->publish(*msg);
}
void Subscriber::recall(sensor_msgs::msg::Image::SharedPtr A)
{
    RCLCPP_INFO(this->get_logger(),"header:\n %d.%d \n%s",A->header.stamp.sec,A->header.stamp.nanosec,A->header.frame_id.c_str());
    RCLCPP_INFO(this->get_logger(),"data:%d \n %s \n %d",A->width,A->encoding.c_str(),A->height);
    mid_save=cv_bridge::toCvShare(A,"bgr8")->image;
    stamp1=A->header.stamp.sec;
    nstamp=A->header.stamp.nanosec;
}
void Subscriber::recall_ser(const std::shared_ptr<std_srvs::srv::Empty::Request> m,const std::shared_ptr<std_srvs::srv::Empty::Response> n)
{
    RCLCPP_INFO(this->get_logger(),"it has been saved");
    std::filesystem::create_directory("images");
    cv::imwrite("images/"+std::to_string(stamp1)+"_"+std::to_string(nstamp)+".png",mid_save);
}
int main(int argc,char **argv)
{
    rclcpp::init(argc,argv);
    auto ImageSubscribeNode=std::make_shared<Subscriber>("sub");
    rclcpp::spin(ImageSubscribeNode);
    rclcpp::shutdown();
    return 0;
}

