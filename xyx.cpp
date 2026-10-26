#include <ros/ros.h>
// PCL specific includes 用PCL的时候需要用到的库
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h> // 添加此行
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>           // 直通滤波头文件
#include <pcl/filters/conditional_removal.h>   // 关键头文件
#include <pcl/segmentation/sac_segmentation.h> // 关键头文件
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
// 先声明发布器，因为回调函数里面会用到
ros::Publisher pub;
ros::Publisher pubsub;
ros::Publisher pubcenter;
ros::Publisher pub_plane;
// 全局变量：存储累积的点云
pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
// const 可以保护input的内容
// sensor_msgs::PointCloud2ConstPtr用来创建常量点云的指针
// sensor_msgs::PointCloud2 用来创建普通的点云容器
// 点云的数据类型还有很多，但是常用这个，参考第4步
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input)
{
    // Step 1: 转换ROS消息到PCL格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input, *cloud);

    // Step 2: 直通滤波（Z/X/Y轴）
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(1, 3);
    pass.filter(*cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-4.0, 4.0);
    pass.filter(*cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(6.0, 10.0);
    pass.filter(*cloud);

    // 累积点云
    *accumulated_cloud += *cloud;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE); // 检测平面模型
    seg.setMethodType(pcl::SAC_RANSAC);    // 使用RANSAC算法
    seg.setDistanceThreshold(0.04);        // 平面内点的距离阈值（单位：米）
    seg.setInputCloud(accumulated_cloud);
    seg.segment(*inliers, *coefficients); // 执行分割

    // Step 2: 提取平面点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(accumulated_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false); // 提取平面（设为true则提取非平面部分）
    extract.filter(*plane_cloud);

    // Step 3: 将平面点云发布为红色
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_plane->width = plane_cloud->size();
    colored_plane->height = 1;
    colored_plane->points.resize(plane_cloud->size());

    for (size_t i = 0; i < plane_cloud->size(); ++i)
    {
        colored_plane->points[i].x = plane_cloud->points[i].x;
        colored_plane->points[i].y = plane_cloud->points[i].y;
        colored_plane->points[i].z = plane_cloud->points[i].z;
        colored_plane->points[i].r = 0; // 红色
        colored_plane->points[i].g = 0;
        colored_plane->points[i].b = 255;
    }

    // 发布平面点云
    sensor_msgs::PointCloud2 output_plane;
    pcl::toROSMsg(*colored_plane, output_plane);
    output_plane.header = input->header;
    pub_plane.publish(output_plane); // 需要提前定义 pub_plane
    // Step 3: 计算质心并提取中心点云
    float min_x = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();

    for (const auto &point : *plane_cloud)
    {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_z = std::min(min_z, point.z);
        max_z = std::max(max_z, point.z);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }

    // 计算XY平面的中心点
    float center_x = (min_x + max_x) / 2.0f;
    float center_z = (min_z + max_z) / 2.0f;
    float center_y = min_y;
    ROS_INFO("center_x:%f,center_z:%f,center_y:%f", center_x, center_z, center_y );
    // 定义立方体范围（质心 ± range 米）
    float range = 0.2; // 1米范围内的点
    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>());

    // X轴条件
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
        new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::GT, center_x - range)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
        new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::LT, center_x + range)));

    // Z轴条件
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
        new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GT, center_z - range)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
        new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, center_z + range)));

    // 应用条件滤波
    pcl::ConditionalRemoval<pcl::PointXYZ> cond_removal;
    cond_removal.setCondition(range_cond);
    cond_removal.setInputCloud(plane_cloud);
    cond_removal.setKeepOrganized(false);

    pcl::PointCloud<pcl::PointXYZ>::Ptr centered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cond_removal.filter(*centered_cloud);

    // Step 4: 发布结果
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header = input->header;
    pub.publish(output);

    sensor_msgs::PointCloud2 output_acc;
    pcl::toROSMsg(*accumulated_cloud, output_acc);
    output_acc.header = input->header;
    pubsub.publish(output_acc);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cloud->width = centered_cloud->size();
    colored_cloud->height = 1;
    colored_cloud->is_dense = false;
    colored_cloud->points.resize(centered_cloud->size());

    // 为每个点设置坐标和颜色（红色）
    for (size_t i = 0; i < centered_cloud->size(); ++i)
    {
        colored_cloud->points[i].x = centered_cloud->points[i].x;
        colored_cloud->points[i].y = centered_cloud->points[i].y;
        colored_cloud->points[i].z = centered_cloud->points[i].z;
        colored_cloud->points[i].r = 255; // R通道=255（红色）
        colored_cloud->points[i].g = 0;   // G通道=0
        colored_cloud->points[i].b = 0;   // B通道=0
    }

    // 发布居中的点云
    sensor_msgs::PointCloud2 output_centered;
    pcl::toROSMsg(*colored_cloud, output_centered);
    output_centered.header = input->header;
    pubcenter.publish(output_centered);
}

int main(int argc, char **argv)
{
    // Initialize ROS
    ros::init(argc, argv, "my_pcl_tutorial");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    // 创建订阅器的时候是和PCL没有关系的，只和ROS有关系，回调函数才需要涉及PCL
    // 这里的Input可以改成自己想订阅的topic,比如"/rslidar_points"
    ros::Subscriber sub = nh.subscribe("/cloud_registered", 1, cloud_cb);

    // Create a ROS publisher for the output point cloud
    // 创建发布器的时候和PCL有关系，因为要指定发布出去的信息格式
    pub = nh.advertise<sensor_msgs::PointCloud2>("output", 1);
    pubsub = nh.advertise<sensor_msgs::PointCloud2>("output_acc", 1);
    pubcenter = nh.advertise<sensor_msgs::PointCloud2>("pubcenter", 1);
    pub_plane = nh.advertise<sensor_msgs::PointCloud2>("pub_plane", 1);
    // Spin
    ros::spin(); // ros::spin()用于调用所有可触发的回调函数,将进入循环,不会返回,类似于在循环里反               复调用spinOnce()
}
