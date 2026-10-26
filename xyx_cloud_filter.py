#!/usr/bin/env python
import rospy
import pcl
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,PointStamped
from collections import deque
import cv2
from scipy.spatial import cKDTree
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

class PointCloudProcessor:
    def __init__(self):
        self.filter_cloud_pub = rospy.Publisher("/cloud_filtered", PointCloud2, queue_size=1)
        rospy.Subscriber("/cloud_registered", PointCloud2, self.callback)
        self.buffer = deque(maxlen=150)  

        self.line_inlier_pub = rospy.Publisher("/line_inliers", PointCloud2, queue_size=1)
        self.line_marker_pub = rospy.Publisher("/fitted_lines_marker", Marker, queue_size=10)
        self.point_pub = rospy.Publisher('/basketball_center_point', PointStamped, queue_size=10)
        self.mag_header = None
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=1)
        self.center = [0,0,0]
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.calculate_pose)


    def calculate_pose(self,msg:Odometry):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        (roll, pitch, yaw) = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        # 坐标转换：将原始 (x, y, z) 转换为 (-y, x, z)
        x1 = -position.y
        y1 = position.x
        z1 = position.z
        transformed_position = (x1, y1, z1)
        x2, y2, z2 = self.center

        # 方向向量（x正向为0°，逆时针为正）
        dx = x2 - x1
        dy = y2 - y1
        target_yaw = math.atan2(dy, dx)  # 单位：弧度

        # 水平距离（不考虑z轴）
        distance = math.hypot(dx, dy)

        # 打印或保存结果
        rospy.loginfo(f"当前坐标: ({x1:.2f}, {y1:.2f}, {z1:.2f})")
        rospy.loginfo(f"目标坐标: ({x2:.2f}, {y2:.2f}, {z2:.2f})")
        rospy.loginfo(f"目标角度（rad）: {target_yaw:.3f}，角度（°）: {math.degrees(target_yaw):.2f}")
        rospy.loginfo(f"水平距离: {distance:.2f}")



    def crop_box_filter(self, cloud, x_range, y_range, z_range):
        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("x")
        passthrough.set_filter_limits(x_range[0], x_range[1])
        cloud = passthrough.filter()

        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("y")
        passthrough.set_filter_limits(y_range[0], y_range[1])
        cloud = passthrough.filter()

        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("z")
        passthrough.set_filter_limits(z_range[0], z_range[1])
        cloud = passthrough.filter()

        return cloud
    
    @staticmethod
    def normalize_vector(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def angle_between_vectors(v1, v2):
        v1_u = PointCloudProcessor.normalize_vector(v1)
        v2_u = PointCloudProcessor.normalize_vector(v2)
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = np.arccos(dot)
        return angle  # 弧度

    def filter_similar_lines(self, lines, angle_threshold_deg=5):
        angle_threshold = np.deg2rad(angle_threshold_deg)
        groups = []
        line_used = [False] * len(lines)

        for i, line_i in enumerate(lines):
            if line_used[i]:
                continue
            vec_i = np.array(line_i[3:6])
            vec_i = self.normalize_vector(vec_i)
            group = [i]
            for j in range(i + 1, len(lines)):
                if line_used[j]:
                    continue
                vec_j = np.array(lines[j][3:6])
                vec_j = self.normalize_vector(vec_j)
                angle = self.angle_between_vectors(vec_i, vec_j)
                # 反向也算相似
                angle = min(angle, np.pi - angle)
                if angle < angle_threshold:
                    group.append(j)
                    line_used[j] = True
            groups.append(group)

        # 每组保留第一个代表直线
        filtered_lines = [lines[g[0]] for g in groups]
        return filtered_lines
    

    def fit_center(self,cloud,x_range, y_range, z_range):

        if cloud is None or cloud.size == 0:
            rospy.logwarn("输入点云为空，跳过处理")
            return

        points = np.asarray(cloud.to_array(), dtype=np.float32)
        
        if points.shape[0] == 0 or points.shape[1] < 3:
            rospy.logwarn("点云格式异常，转换后为空")
            return
        # 去除内矩形计算
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        z_vals = points[:, 2]

        # 按比例选取 20% ~ 80% 之间的值作为内矩形边界
        x_min, x_max = np.percentile(x_vals, [20, 80])
        y_min, y_max = np.percentile(y_vals, [20, 80])
        z_min, z_max = np.percentile(z_vals, [20, 80])

        if x_max - x_min < 1e-3 or z_max - z_min < 1e-3:
            rospy.logwarn("内矩形边界过小，可能异常")
            return

        # 构造内矩形范围
        x_range_inner = (x_min, x_max)
        y_range_inner = (y_min, y_max)
        z_range_inner = (z_min, z_max)
        # 获取立方体内的点云
        inside_cloud = self.crop_box_filter(cloud, x_range_inner, y_range_inner, z_range_inner)

        if inside_cloud is None or inside_cloud.size == 0:
            rospy.logwarn("裁剪后点云为空，跳过拟合")
            return

        # 转为 ndarray 便于处理
        all_points = cloud.to_array()
        inside_points = inside_cloud.to_array()

        # 构造外部点：即不在 inside_points 中的点
        # 由于浮点数比较需考虑精度，用行匹配实现
        inside_set = set(map(tuple, inside_points))
        outer_points = np.array([pt for pt in all_points if tuple(pt) not in inside_set], dtype=np.float32)

        ros_cloud = pc2.create_cloud_xyz32(self.mag_header, outer_points)
        self.filter_cloud_pub.publish(ros_cloud)   

        # 计算中心点（假设固定y轴）
        # center = self.fit_rectangle_center_fixed_y(outer_points)

        # 计算四边形顶点
        corners = self.fit_rectangle_from_pointcloud(outer_points)
        # 判断角点是否有效（必须是4个，且不能全为0）
        if not corners or len(corners) != 4:
            rospy.logwarn("角点数量不为4，跳过该帧")
            return

        corners_np = np.array(corners, dtype=np.float32)

        # 检查是否是异常点（例如全部为(0,0,0)）
        if np.allclose(corners_np, 0):
            rospy.logwarn("角点数据为零点，疑似拟合失败")
            return
        # 转换为 numpy 数组，方便向量化计算
        corners_np = np.array(corners)  # shape: (4, 3)
    
        # 按照排列顺序：右上、左上、左下、右下
        right_top, left_top, left_bottom, right_bottom = corners_np
        
        # 根据你的代码，A=右上，B=左上，C=左下，D=右下
        A = right_top     # 右上
        B = left_top      # 左上  
        C = left_bottom   # 左下
        D = right_bottom  # 右下
        
        
        # 向量 AB 和 AD
        vec_ab = B - A  # 从右上到左上
        vec_ad = D - A  # 从右上到右下
        
        
        # 法向量 = AB x AD
        normal_vec = np.cross(vec_ab, vec_ad)
        normal_vec = normal_vec / np.linalg.norm(normal_vec)  # 单位向量
        
        
        # 底边中点 (右下和右上的中点)
        bottom_mid = (C + D) / 2.0
        print(f"底边中点 (右上+右下)/2: {bottom_mid}")
        
        # 偏移：前方22cm + z方向向下10cm
        offset_vec = normal_vec * 0.325
        print(f"偏移量:{offset_vec}")
        


        hoop_center = bottom_mid + offset_vec
        basket_center = tuple(hoop_center)
        print(f"球框中心:{basket_center}")

        self.center =  basket_center
        # 计算中心点
        center = np.mean(corners_np, axis=0)  # shape: (3,)
        print(f"中心点: x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}")
        print(f"边界:{corners}")
        # print(f"中心点{center}")
        # 发布Marker可视化（header来自回调消息）
        self.publish_rectangle_marker(corners, basket_center,marker_id=0)



    def fit_rectangle_center_fixed_y(self, points, outlier_threshold=0.1):
        """
        拟合 outer_points 的矩形中心点，假设 y 坐标固定（即只在 x-z 平面分布）
        使用分位数方法去除噪点
        :param points: np.ndarray (N, 3)
        :param outlier_threshold: 去除异常值的阈值（0-0.5之间）
        :return: center_point: np.ndarray (1, 3)
        """

        points = np.array(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            rospy.logwarn("点云数据格式不正确，shape=%s" % str(points.shape))
            return []
        x_vals = points[:, 0]
        y_vals = points[:,1]
        z_vals = points[:, 2]
        
        # 使用分位数方法去除噪点
        x_min = np.percentile(x_vals, outlier_threshold * 100)
        x_max = np.percentile(x_vals, (1 - outlier_threshold) * 100)
        z_min = np.percentile(z_vals, outlier_threshold * 100)
        z_max = np.percentile(z_vals, (1 - outlier_threshold) * 100)
        

        x_center = (x_min + x_max) / 2
        z_center = (z_min + z_max) / 2
        

        # y 为固定值，取均值
        y_center = np.mean(points[:, 1])
        
        return np.array([x_center, y_center, z_center], dtype=np.float32)

    def get_rectangle_corners_fixed_y(self, points, outlier_threshold=0.1):
        """
        获取矩形的四个角点，使用分位数方法去除噪点
        :param points: np.ndarray (N, 3)
        :param outlier_threshold: 去除异常值的阈值（0-0.5之间）
        :return: corners: list of tuples
        """
        x_vals = points[:, 0]
        z_vals = points[:, 2]
        
        # 使用分位数方法去除噪点
        x_min = np.percentile(x_vals, outlier_threshold * 100)
        x_max = np.percentile(x_vals, (1 - outlier_threshold) * 100)
        z_min = np.percentile(z_vals, outlier_threshold * 100)
        z_max = np.percentile(z_vals, (1 - outlier_threshold) * 100)
        
        y_mean = np.mean(points[:, 1])
        
        corners = [
            (x_min, y_mean, z_min),
            (x_max, y_mean, z_min),
            (x_max, y_mean, z_max),
            (x_min, y_mean, z_max)
        ]
        return corners
    
    def fit_rectangle_from_pointcloud(self, points: np.ndarray):
        points = np.array(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            rospy.logwarn("点云数据格式不正确，shape=%s" % str(points.shape))
            return []

        xz_points = points[:, [0, 2]]  # 拟合用XZ
        origin_points = points         # 保存原始3D点用于还原Y

        # 剔除异常值
        def filter_percentile(p, low=0.05, high=0.95):
            x_min, x_max = np.percentile(p[:, 0], [low*100, high*100])
            z_min, z_max = np.percentile(p[:, 1], [(low+0.1)*100, (high-0.05)*100])
            mask = (p[:, 0] >= x_min) & (p[:, 0] <= x_max) & \
                (p[:, 1] >= z_min) & (p[:, 1] <= z_max)
            return p[mask]
        
        xz_points_filtered = filter_percentile(xz_points)
        if len(xz_points_filtered) < 50:
            rospy.logwarn("点数量太少，无法拟合矩形")
            return []

        rect = cv2.minAreaRect(xz_points_filtered)
        box = cv2.boxPoints(rect)  # shape: (4, 2), in XZ
        box = self.reorder_box_points_2d(np.array(box))

        # 还原Y：从原始点云中寻找最近的点来获取Y
        tree = cKDTree(points[:, [0, 2]])
        _, indices = tree.query(box, k=1)  # 每个角点找最近的点
        box_3d = [tuple(points[i]) for i in indices]  # 直接用原始点

        return box_3d



    def reorder_box_points_2d(self,box_2d):
        """
        重新排列2D边界框的角点 (XZ平面)，按照指定顺序排列
        
        期望顺序：右上、左上、左下、右下 (在XZ平面上)
        
        Args:
            box_2d: 4个2D角点的numpy数组，shape为(4, 2)，坐标为(x, z)
        
        Returns:
            重新排序后的2D角点数组，shape为(4, 2)
        """
        points = np.array(box_2d)
        
        # 计算中心点
        x_center = np.mean(points[:, 0])
        z_center = np.mean(points[:, 1])  # 注意：这里是z坐标，在第1维
        
        # 找到每个象限的点
        right_high_idx = None  # 右上 (x>center, z>center)
        left_high_idx = None   # 左上 (x<center, z>center)  
        left_low_idx = None    # 左下 (x<center, z<center)
        right_low_idx = None   # 右下 (x>center, z<center)
        
        for i, point in enumerate(points):
            x, z = point
            
            if x > x_center and z > z_center:
                right_high_idx = i
            elif x <= x_center and z > z_center:
                left_high_idx = i
            elif x <= x_center and z <= z_center:
                left_low_idx = i
            elif x > x_center and z <= z_center:
                right_low_idx = i
        
        # 按照目标顺序排列：右上、左上、左下、右下
        ordered_indices = [right_high_idx, left_high_idx, left_low_idx, right_low_idx]
        
        
        reordered_box = points[ordered_indices]
        return reordered_box

    def callback(self, msg):
        # points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points_list =[(-y, x, z) for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)]

        if len(points_list) == 0:
            rospy.logwarn("收到空点云")
            return
        self.mag_header = msg.header
        points_np = np.array(points_list, dtype=np.float32)
        self.buffer.append(points_np)

        if len(self.buffer) < 1:
            return

        all_points = np.vstack(self.buffer)

        cloud = pcl.PointCloud()
        cloud.from_array(all_points)
        x_range=(0, 5)
        y_range=(5, 7)
        z_range=(0.3, 2)
        cropped = self.crop_box_filter(cloud, x_range,y_range,z_range)
        self.fit_center(cropped,x_range,y_range,z_range)

        # ros_cloud = pc2.create_cloud_xyz32(self.mag_header, cropped.to_array().tolist())
        # self.filter_cloud_pub.publish(ros_cloud)   


    def publish_rectangle_marker(self,corners, center, marker_id=0):
        """
        使用传入的header（包含frame_id和stamp）发布矩形Marker
        :param pub: ROS Publisher
        :param corners: 四个顶点列表 [(x,y,z), ...]
        :param center: 中心点 (x,y,z)
        :param header: std_msgs/Header，包含frame_id和stamp
        :param marker_id: int，Marker的ID
        """
        # 矩形边界框 (LINE_STRIP)
        marker_box = Marker()
        marker_box.header = self.mag_header
        marker_box.ns = "rectangle"
        marker_box.id = marker_id
        marker_box.type = Marker.LINE_STRIP
        marker_box.action = Marker.ADD
        marker_box.scale.x = 0.02
        marker_box.color.r = 0.0
        marker_box.color.g = 1.0
        marker_box.color.b = 0.0
        marker_box.color.a = 1.0
        marker_box.points = [Point(*c) for c in corners]
        marker_box.points.append(Point(*corners[0]))  # 闭合

        # 对角线 (LINE_LIST)
        marker_diag = Marker()
        marker_diag.header = self.mag_header
        marker_diag.ns = "rectangle"
        marker_diag.id = marker_id+1
        marker_diag.type = Marker.LINE_LIST
        marker_diag.action = Marker.ADD
        marker_diag.scale.x = 0.01
        marker_diag.color.r = 1.0
        marker_diag.color.g = 0.0
        marker_diag.color.b = 0.0
        marker_diag.color.a = 1.0

        marker_diag.points.append(Point(*corners[0]))
        marker_diag.points.append(Point(*corners[2]))
        marker_diag.points.append(Point(*corners[1]))
        marker_diag.points.append(Point(*corners[3]))

        # 中心点 (SPHERE)
        marker_center = Marker()
        marker_center.header = self.mag_header
        marker_center.ns = "rectangle"
        marker_center.id = marker_id+2
        marker_center.type = Marker.SPHERE
        marker_center.action = Marker.ADD
        marker_center.pose.position.x = center[0]
        marker_center.pose.position.y = center[1]
        marker_center.pose.position.z = center[2]
        marker_center.pose.orientation.x = 0.0
        marker_center.pose.orientation.y = 0.0
        marker_center.pose.orientation.z = 0.0
        marker_center.pose.orientation.w = 1.0

        marker_center.scale.x = 0.05
        marker_center.scale.y = 0.05
        marker_center.scale.z = 0.05
        marker_center.color.r = 0.0
        marker_center.color.g = 0.0
        marker_center.color.b = 1.0
        marker_center.color.a = 1.0

        # 发布
        self.marker_pub.publish(marker_box)
        self.marker_pub.publish(marker_diag)
        self.marker_pub.publish(marker_center)

if __name__ == '__main__':
    rospy.init_node("cloud_cropper_node", anonymous=True)
    processor = PointCloudProcessor()
    rospy.spin()
