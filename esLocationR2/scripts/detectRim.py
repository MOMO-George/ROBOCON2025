#!/usr/bin/env python
import rospy
import pcl
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
from collections import deque
import cv2
from scipy.spatial import cKDTree
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
from port import SerialPorter
from std_msgs.msg import Int32


class PointCloudBasketDetector:
    """
    独立的点云篮球框识别类
    """
    
    def __init__(self, outlier_threshold=0.1):
        """
        初始化检测器
        
        Args:
            outlier_threshold: 去除异常值的阈值（0-0.5之间）
        """
        self.outlier_threshold = outlier_threshold
    
    def crop_box_filter(self, cloud, x_range, y_range, z_range):
        """
        使用通过滤波器裁剪点云到指定范围
        """
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
    
    def extract_outer_points(self, cloud, inner_percentiles=(25, 85, 30, 80)):
        """
        提取外边界点，去除内部指定百分位范围的点
        
        Args:
            cloud: pcl.PointCloud对象
            inner_percentiles: (x_low, x_high, z_low, z_high) 内部区域的百分位数
            
        Returns:
            outer_points: numpy数组，外边界点
        """
        if cloud is None or cloud.size == 0:
            rospy.logwarn("输入点云为空")
            return np.array([])

        points = np.asarray(cloud.to_array(), dtype=np.float32)
        
        if points.shape[0] == 0 or points.shape[1] < 3:
            rospy.logwarn("点云格式异常，转换后为空")
            return np.array([])

        # 按比例选取内矩形边界
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        z_vals = points[:, 2]

        x_min, x_max = np.percentile(x_vals, [inner_percentiles[0], inner_percentiles[1]])
        y_min, y_max = np.percentile(y_vals, [inner_percentiles[0], inner_percentiles[1]])
        z_min, z_max = np.percentile(z_vals, [inner_percentiles[2], inner_percentiles[3]])

        if x_max - x_min < 1e-3 or z_max - z_min < 1e-3:
            rospy.logwarn("内矩形边界过小，可能异常")
            return np.array([])

        # 构造内矩形范围
        x_range_inner = (x_min, x_max)
        y_range_inner = (y_min, y_max)
        z_range_inner = (z_min, z_max)
        
        # 获取立方体内的点云
        inside_cloud = self.crop_box_filter(cloud, x_range_inner, y_range_inner, z_range_inner)

        if inside_cloud is None or inside_cloud.size == 0:
            rospy.logwarn("裁剪后点云为空")
            return np.array([])

        # 转为 ndarray 便于处理
        all_points = cloud.to_array()
        inside_points = inside_cloud.to_array()

        # 构造外部点：即不在 inside_points 中的点
        inside_set = set(map(tuple, inside_points))
        outer_points = np.array([pt for pt in all_points if tuple(pt) not in inside_set], dtype=np.float32)
        
        return outer_points
    
    def fit_rectangle_from_pointcloud(self, points):
        """
        从点云拟合矩形角点
        """
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

    def reorder_box_points_2d(self, box_2d):
        """
        重新排列2D边界框的角点 (XZ平面)，按照指定顺序排列
        期望顺序：右上、左上、左下、右下 (在XZ平面上)
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
    
    def calculate_basket_center(self, corners, offset_distance=0.325):
        """
        根据矩形角点计算篮球框中心位置
        """
        if not corners or len(corners) != 4:
            rospy.logwarn("角点数量不为4，无法计算篮球框中心")
            return None

        corners_np = np.array(corners, dtype=np.float32)

        # 检查是否是异常点（例如全部为(0,0,0)）
        if np.allclose(corners_np, 0):
            rospy.logwarn("角点数据为零点，疑似拟合失败")
            return None

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
        
        # 底边中点 (左下和右下的中点)
        bottom_mid = (C + D) / 2.0
        print(f"底边中点 (左下+右下)/2: {bottom_mid}")
        
        # 偏移：前方offset_distance
        offset_vec = normal_vec * offset_distance
        print(f"偏移量:{offset_vec}")
        
        hoop_center = bottom_mid + offset_vec
        basket_center = tuple(hoop_center)
        print(f"球框中心:{basket_center}")
        
        return basket_center
    
    def detect_basket(self, cloud, detection_range=None):
        """
        检测篮球框位置的主要方法
        """
        # 默认检测范围
        if detection_range is None:
            detection_range = {
                'x': (1, 5),
                'y': (5, 7),
                'z': (0.5, 2)
            }
        
        result = {
            'success': False,
            'basket_center': None,
            'corners': [],
            'outer_points': np.array([])
        }
        
        try:
            # 如果输入是numpy数组，转换为pcl点云
            if isinstance(cloud, np.ndarray):
                pcl_cloud = pcl.PointCloud()
                pcl_cloud.from_array(cloud.astype(np.float32))
                cloud = pcl_cloud
            
            # 裁剪到检测范围
            cropped_cloud = self.crop_box_filter(
                cloud, 
                detection_range['x'],
                detection_range['y'], 
                detection_range['z']
            )
            
            if cropped_cloud is None or cropped_cloud.size == 0:
                rospy.logwarn("裁剪后点云为空")
                return result
            
            # 提取外边界点
            outer_points = self.extract_outer_points(cropped_cloud)
            if len(outer_points) == 0:
                return result
            
            result['outer_points'] = outer_points
            
            # 拟合矩形角点
            corners = self.fit_rectangle_from_pointcloud(outer_points)
            if not corners or len(corners) != 4:
                rospy.logwarn("无法获取有效的矩形角点")
                return result
            
            result['corners'] = corners
            
            # 计算篮球框中心
            basket_center = self.calculate_basket_center(corners)
            if basket_center is None:
                return result
            
            result['basket_center'] = basket_center
            result['success'] = True
            
            return result
            
        except Exception as e:
            rospy.logerr(f"篮球框检测过程中发生错误: {e}")
            return result


class NavigationController:
    """
    导航控制类，处理位置计算和导航逻辑
    """
    
    def __init__(self, serial_port="/dev/serial_send", self_offset=[0.0, 0.3]):
        """
        初始化导航控制器
        
        Args:
            serial_port: 串口设备路径
            self_offset: 自身偏移量 [x, y]
        """
        self.selfOffset = self_offset
        self.target_center = [0, 0, 0]
        
        # 初始化串口
        try:
            self.port = SerialPorter(serial_port)
            self.port.open_port()
            self.port_available = True
        except Exception as e:
            rospy.logwarn(f"无法打开串口: {e}")
            self.port_available = False
    
    def set_target_center(self, center):
        """
        设置目标中心点
        
        Args:
            center: 目标中心坐标 [x, y, z]
        """
        self.target_center = center
    
    def calculate_navigation_data(self, odometry_msg):
        """
        根据里程计数据计算导航信息
        
        Args:
            odometry_msg: nav_msgs/Odometry消息
            
        Returns:
            dict: 包含导航信息的字典
        """
        position = odometry_msg.pose.pose.position
        orientation = odometry_msg.pose.pose.orientation

        (roll, pitch, yaw) = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        # 坐标转换：将原始 (x, y, z) 转换为 (-y, x, z)
        x1 = -position.y - self.selfOffset[1] * np.sin(yaw)
        y1 = position.x + self.selfOffset[1] * np.cos(yaw)
        z1 = position.z
        current_position = (x1, y1, z1)
        
        x2, y2, z2 = self.target_center

        # 方向向量（x正向为0°，逆时针为正）
        dx = x2 - x1
        dy = y2 - y1
        target_yaw = math.pi / 2 - math.atan2(dy, dx)  # 单位：弧度
        turn_angle = -yaw - target_yaw
        
        # 水平距离（不考虑z轴）
        distance = math.hypot(dx, dy) - 0.3
        
        navigation_data = {
            'current_position': current_position,
            'target_position': (x2, y2, z2),
            'current_yaw': yaw,
            'target_yaw': target_yaw,
            'turn_angle': turn_angle,
            'distance': distance
        }
        
        return navigation_data
    
    def send_navigation_command(self, navigation_data):
        """
        发送导航命令到串口
        
        Args:
            navigation_data: 导航数据字典
        """
        if not self.port_available:
            return
        
        try:
            x1, y1, z1 = navigation_data['current_position']
            x2, y2, z2 = navigation_data['target_position']
            turn_angle = navigation_data['turn_angle']
            distance = navigation_data['distance']
            
            self.port.send_data([x1, y1, x2, y2, turn_angle, distance])
        except Exception as e:
            rospy.logerr(f"发送导航命令失败: {e}")
    
    def log_navigation_info(self, navigation_data):
        """
        打印导航信息
        """
        x1, y1, z1 = navigation_data['current_position']
        x2, y2, z2 = navigation_data['target_position']
        target_yaw = navigation_data['target_yaw']
        current_yaw = navigation_data['current_yaw']
        turn_angle = navigation_data['turn_angle']
        distance = navigation_data['distance']
        
        rospy.loginfo(f"当前坐标: ({x1:.2f}, {y1:.2f}, {z1:.2f})")
        rospy.loginfo(f"目标坐标: ({x2:.2f}, {y2:.2f}, {z2:.2f})")
        rospy.loginfo(f"目标角度: {math.degrees(target_yaw):.3f}，角度（°）: {math.degrees(current_yaw):.2f},还要转的角度{math.degrees(turn_angle):.2f}")
        rospy.loginfo(f"水平距离: {distance:.2f}")
    
    def close(self):
        """
        关闭串口连接
        """
        if self.port_available:
            try:
                self.port.close_port()
            except Exception as e:
                rospy.logerr(f"关闭串口失败: {e}")


class VisualizationPublisher:
    """
    可视化发布类，处理RViz标记发布
    """
    
    def __init__(self):
        """
        初始化可视化发布器
        """
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=1)
    
    def publish_rectangle_marker(self, corners, center, header, marker_id=0):
        """
        发布矩形标记到RViz
        
        Args:
            corners: 四个顶点列表 [(x,y,z), ...]
            center: 中心点 (x,y,z)
            header: std_msgs/Header，包含frame_id和stamp
            marker_id: int，Marker的ID
        """
        # 矩形边界框 (LINE_STRIP)
        marker_box = Marker()
        marker_box.header = header
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
        marker_diag.header = header
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
        marker_center.header = header
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

        # 发布所有标记
        self.marker_pub.publish(marker_box)
        self.marker_pub.publish(marker_diag)
        self.marker_pub.publish(marker_center)


class PointCloudProcessor:
    """
    主要的点云处理ROS节点类
    """
    
    def __init__(self):
        """
        初始化点云处理器
        """
        # ROS发布器和订阅器
        self.filter_cloud_pub = rospy.Publisher("/cloud_filtered", PointCloud2, queue_size=1)
        self.point_pub = rospy.Publisher('/basketball_center_point', PointStamped, queue_size=10)
        
        # 点云缓冲区
        self.buffer = deque(maxlen=300)
        self.mag_header = None
        
        # 创建各个功能模块
        self.detector = PointCloudBasketDetector()
        self.navigator = NavigationController()
        self.visualizer = VisualizationPublisher()
        
        # ROS订阅器
        rospy.Subscriber("/cloud_registered", PointCloud2, self.pointcloud_callback)
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.odometry_callback)
        
        # 检测参数配置
        self.detection_range = {
            'x': (1, 5),
            'y': (5, 7),
            'z': (0.5, 2)
        }
    
    def pointcloud_callback(self, msg):
        """
        点云数据回调函数
        
        Args:
            msg: sensor_msgs/PointCloud2消息
        """
        # 坐标转换并读取点云
        points_list = [(-y, x, z) for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)]

        if len(points_list) == 0:
            rospy.logwarn("收到空点云")
            return
        
        self.mag_header = msg.header
        points_np = np.array(points_list, dtype=np.float32)
        self.buffer.append(points_np)

        if len(self.buffer) < 1:
            return

        # 合并缓冲区中的点云
        all_points = np.vstack(self.buffer)

        # 使用检测器进行篮球框检测
        result = self.detector.detect_basket(all_points, self.detection_range)
        
        if result['success']:
            # 更新导航目标
            self.navigator.set_target_center(result['basket_center'])
            
            # 发布滤波后的点云
            if len(result['outer_points']) > 0:
                ros_cloud = pc2.create_cloud_xyz32(self.mag_header, result['outer_points'])
                self.filter_cloud_pub.publish(ros_cloud)
            
            # 发布可视化标记
            self.visualizer.publish_rectangle_marker(
                result['corners'], 
                result['basket_center'], 
                self.mag_header,
                marker_id=0
            )
            
            print(f"检测成功！篮球框中心: {result['basket_center']}")
            print(f"矩形角点: {result['corners']}")
        else:
            rospy.logwarn("篮球框检测失败")
    
    def odometry_callback(self, msg):
        """
        里程计数据回调函数
        
        Args:
            msg: nav_msgs/Odometry消息
        """
        # 计算导航数据
        navigation_data = self.navigator.calculate_navigation_data(msg)
        
        # 发送导航命令
        self.navigator.send_navigation_command(navigation_data)
        
        # 打印导航信息
        self.navigator.log_navigation_info(navigation_data)
    
    def shutdown(self):
        """
        节点关闭清理函数
        """
        self.navigator.close()
        rospy.loginfo("点云处理节点已关闭")


if __name__ == '__main__':
    try:
        rospy.init_node("cloud_cropper_node", anonymous=True)
        processor = PointCloudProcessor()
        
        # 注册关闭回调
        rospy.on_shutdown(processor.shutdown)
        
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"节点运行错误: {e}")
    finally:
        rospy.loginfo("程序退出")