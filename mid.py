#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
import open3d as o3d
from std_msgs.msg import Header

class PlaneDetector:
    def __init__(self):
        rospy.init_node('plane_detector', anonymous=True)
        
        # 订阅点云话题（根据您的实际话题修改）
        self.cloud_sub = rospy.Subscriber("/point_cloud_topic", PointCloud2, self.cloud_callback)
        
        # 发布检测到的平面中心点
        self.center_pub = rospy.Publisher("/plane_center", PointStamped, queue_size=1)
        
        # 定义裁剪区域 (x,y,z)
        self.x_range = [7.5, 8.0]    # 7.5m到8m
        self.y_range = [-6.0, -2.0]   # -6m到-2m
        self.z_range = [2.4, 3.6]     # 2.4m到3.6m
        
        # 目标平面参数
        self.plane_size = 2.0         # 2m×2m平面
        self.hole_size = 1.0          # 中心1m×1m空洞

    def cloud_callback(self, msg):
        try:
            # 将ROS PointCloud2转换为Open3D点云
            pcd = self.ros_to_o3d(msg)
            
            # 裁剪到指定区域
            cropped_pcd = self.crop_point_cloud(pcd)
            
            if len(cropped_pcd.points) < 3:
                rospy.logwarn("裁剪后点云点数不足")
                return
            
            # 检测平面
            plane_center, plane_eq, inliers = self.detect_plane_with_hole(cropped_pcd)
            
            if plane_center is not None:
                # 发布中心点
                self.publish_center(plane_center, msg.header)
                
        except Exception as e:
            rospy.logerr(f"处理点云时出错: {str(e)}")

    def ros_to_o3d(self, ros_cloud):
        """将ROS PointCloud2转换为Open3D点云"""
        points = []
        for p in pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def crop_point_cloud(self, pcd):
        """按指定空间范围裁剪点云"""
        points = np.asarray(pcd.points)
        
        # 创建裁剪掩码
        mask = (points[:,0] >= self.x_range[0]) & (points[:,0] <= self.x_range[1]) & \
               (points[:,1] >= self.y_range[0]) & (points[:,1] <= self.y_range[1]) & \
               (points[:,2] >= self.z_range[0]) & (points[:,2] <= self.z_range[1])
        
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(points[mask])
        return cropped_pcd

    def detect_plane_with_hole(self, pcd):
        """
        检测2m×2m平面并排除中心1m×1m区域
        返回剩余平面的中心点
        """
        # 1. 使用RANSAC检测主平面
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < 100:  # 平面点数太少
            return None, None, None
        
        # 2. 提取平面点云
        plane_cloud = pcd.select_by_index(inliers)
        plane_points = np.asarray(plane_cloud.points)
        
        # 3. 计算平面边界
        min_pt = np.min(plane_points, axis=0)
        max_pt = np.max(plane_points, axis=0)
        plane_dimensions = max_pt - min_pt
        
        # 检查平面尺寸是否接近2m×2m（考虑一定容差）
        if plane_dimensions[0] < 1.8 or plane_dimensions[1] < 1.8:
            rospy.logwarn(f"检测到的平面尺寸不足: {plane_dimensions[0]:.2f}×{plane_dimensions[1]:.2f}m")
            return None, None, None
        
        # 4. 计算平面中心区域（要排除的1m×1m区域）
        plane_center = np.mean(plane_points, axis=0)
        hole_min = plane_center - np.array([self.hole_size/2, self.hole_size/2, 0])
        hole_max = plane_center + np.array([self.hole_size/2, self.hole_size/2, 0])
        
        # 5. 排除中心区域点
        hole_mask = (plane_points[:,0] >= hole_min[0]) & \
                   (plane_points[:,0] <= hole_max[0]) & \
                   (plane_points[:,1] >= hole_min[1]) & \
                   (plane_points[:,1] <= hole_max[1])
        
        remaining_points = plane_points[~hole_mask]
        
        if len(remaining_points) < 10:  # 剩余点数太少
            return None, None, None
        
        # 6. 计算剩余平面的中心
        remaining_center = np.mean(remaining_points, axis=0)
        
        return remaining_center, plane_model, inliers

    def publish_center(self, center, header):
        """发布平面中心点"""
        point_msg = PointStamped()
        point_msg.header = header
        point_msg.header.stamp = rospy.Time.now()
        point_msg.point.x = center[0]
        point_msg.point.y = center[1]
        point_msg.point.z = center[2]
        
        self.center_pub.publish(point_msg)
        rospy.loginfo(f"发布平面中心点: x={center[0]:.2f}, y={center[1]:.2f}, z={center[2]:.2f}")

if __name__ == '__main__':
    try:
        detector = PlaneDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass