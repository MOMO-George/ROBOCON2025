#!/usr/bin/env python3
"""
全场篮球定位系统

基于点云数据进行篮板检测和定位的全场版本。
相比近场版本，增加了启动模式以处理远距离时点云稀少的问题。

"""

# 标准库导入
import math
import time
from collections import deque
from typing import List, Tuple, Optional, Union

# 第三方库导入
import cv2
import numpy as np
import pcl
from scipy.spatial import cKDTree

# ROS相关导入
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker

# 本地导入
from port import SerialPorter


class FullCourtLocationConfig:
    """全场定位配置常量"""
    
    # 全场定位范围 (原始中场为 x:(-2,2), y:(5,7))
    FULL_COURT_X_RANGE = (1, 6) #靠zuobian
    FULL_COURT_Y_RANGE = (12.5, 14.5)
    FULL_COURT_Z_RANGE = (1, 2.7)
    
    # 标准篮板尺寸
    STANDARD_BACKBOARD_WIDTH = 1.8   # 米
    STANDARD_BACKBOARD_HEIGHT = 1.05  # 米
    BACKBOARD_TOLERANCE = 0.1  # 尺寸容差
    
    # 启动模式参数
    STARTUP_POINT_THRESHOLD = 1000  # 启动阶段需要累积的最小点数
    STARTUP_BUFFER_SIZE = 50      # 启动阶段缓冲区大小
    MAIN_BUFFER_SIZE = 300        # 主缓冲区大小
    
    # 点云处理参数
    MIN_SINGLE_FRAME_POINTS = 20  # 单帧最小点数
    MIN_PRECHECK_POINTS = 40      # 预检查最小点数
    MIN_OUTER_POINTS = 12         # 外部点最小数量
    MIN_FILTERED_POINTS = 20      # 过滤后最小点数
    
    # 篮框偏移参数
    HOOP_OFFSET_DISTANCE = 0.375  # 篮框距离篮板的偏移(米)
    POSITION_OFFSET_X = 0.2012
    POSITION_OFFSET_Y = -0.2361


class FullCourtPointCloudProcessor:
    """全场篮球定位点云处理器"""
    
    def __init__(self):
        """初始化全场定位处理器"""
        
        # 基本状态
        self.mag_header = None
        self.center = [0, 0, 0]
        self.corners_cjkc = None
        self.use_auto_bbox = False
        
        # 点云缓冲区
        self.buffer = deque(maxlen=FullCourtLocationConfig.MAIN_BUFFER_SIZE)
        
        # 全场定位专用状态
        self.startup_mode = True
        self.accumulated_points = 0
        self.startup_buffer = deque(maxlen=FullCourtLocationConfig.STARTUP_BUFFER_SIZE)
        
        # 初始化ROS接口
        self._init_ros_interface()
        
        # 串口通信（可选）
        # self.port = SerialPorter("/dev/serial_send")
        # self.port.open_port()
        
        # 启动日志
        rospy.loginfo("🚀 全场定位模式启动")
        rospy.loginfo(f"启动阶段将累积点云直到达到 {FullCourtLocationConfig.STARTUP_POINT_THRESHOLD} 个点")

    def _init_ros_interface(self):
        """初始化ROS发布者和订阅者"""
        
        # 订阅者
        rospy.Subscriber("/cloud_registered", PointCloud2, self.callback)
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.calculate_pose)
        
        # 发布者
        self.filter_cloud_pub = rospy.Publisher("/cloud_filtered", PointCloud2, queue_size=1)
        self.line_inlier_pub = rospy.Publisher("/line_inliers", PointCloud2, queue_size=1)
        self.line_marker_pub = rospy.Publisher("/fitted_lines_marker", Marker, queue_size=10)
        self.point_pub = rospy.Publisher('/basketball_center_point', PointStamped, queue_size=10)
        self.plot_pub = rospy.Publisher("/plot", Float32MultiArray, queue_size=10)
        self.data_pub = rospy.Publisher('/controller', Float32MultiArray, queue_size=10)
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=1)

    # ======================== 主要回调函数 ========================
    
    def callback(self, msg: PointCloud2):
        """点云数据回调函数"""
        
        # 读取点云数据
        points_list = [(x, y, z) for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)]
        
        if len(points_list) == 0:
            rospy.logwarn("收到空点云")
            return

        # 转换为numpy数组并创建PCL点云
        points_np = np.array(points_list, dtype=np.float32)
        cloud = pcl.PointCloud()
        cloud.from_array(points_np)

        # 确定裁剪范围
        x_range, y_range, z_range = self._get_crop_ranges()
        
        # 裁剪点云
        cropped = self.crop_box_filter(cloud, x_range, y_range, z_range)
        
        if cropped.size == 0:
            rospy.logwarn("裁剪后无有效点，忽略该帧")
            return
        
        cropped_np = cropped.to_array()
        rospy.loginfo(f"裁剪后点云数量: {cropped_np.shape[0]}")

        # 根据模式处理点云
        if self.startup_mode:
            self._handle_startup_mode(cropped, msg.header)
        else:
            self._handle_normal_mode(cropped, msg.header, x_range, y_range, z_range)

    def calculate_pose(self, msg: Odometry):
        """位姿计算回调函数"""
        
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # 获取偏航角
        (roll, pitch, yaw) = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        
        # 当前位置
        x1, y1, z1 = position.x, position.y, position.z
        
        # 目标位置（篮框中心）
        x2, y2, z2 = self.center
        
        # 计算方向和距离
        dx = x2 - x1
        dy = y2 - y1
        target_yaw = math.pi / 2 - math.atan2(dy, dx)
        turn_angle = -yaw - target_yaw
        distance = math.hypot(dx, dy) - 0.308

        # 发布控制数据
        pub_msg = Float32MultiArray()
        pub_msg.layout = MultiArrayLayout()
        pub_msg.data = [x1, y1, x2, y2, turn_angle, distance]
        
        self.data_pub.publish(pub_msg)
        self.plot_pub.publish(pub_msg)
        
        # 日志输出
        rospy.loginfo(f"当前坐标: ({x1:.2f}, {y1:.2f}, {z1:.2f})")
        rospy.loginfo(f"目标坐标: ({x2:.2f}, {y2:.2f}, {z2:.2f})")
        rospy.loginfo(f"目标角度: {target_yaw:.3f}，当前角度（°）: {math.degrees(yaw):.2f}")
        rospy.loginfo(f"需要转向角度: {math.degrees(turn_angle):.2f}°，距离: {distance:.4f}m")

    # ======================== 模式处理函数 ========================
    
    def _handle_startup_mode(self, cropped: pcl.PointCloud, header):
        """处理启动模式逻辑"""
        
        cropped_np = cropped.to_array()
        self.startup_buffer.append(cropped)
        self.accumulated_points += cropped_np.shape[0]
        
        rospy.loginfo(f"🌱 启动模式：累积点云 {self.accumulated_points}/{FullCourtLocationConfig.STARTUP_POINT_THRESHOLD}")
        
        # 检查是否达到启动阈值
        if self.accumulated_points >= FullCourtLocationConfig.STARTUP_POINT_THRESHOLD:
            rospy.loginfo(f"✅ 启动模式完成！累积了 {self.accumulated_points} 个点，切换到正常模式")
            self.startup_mode = False
            
            # 将启动阶段累积的点云合并到主buffer
            for startup_cloud in self.startup_buffer:
                self.buffer.append(startup_cloud)
            self.startup_buffer.clear()
            
            # 立即进行一次拟合
            self.mag_header = header
            x_range, y_range, z_range = self._get_crop_ranges()
            self._process_accumulated_clouds(x_range, y_range, z_range)

    def _handle_normal_mode(self, cropped: pcl.PointCloud, header, x_range: Tuple, y_range: Tuple, z_range: Tuple):
        """处理正常模式逻辑"""
        
        cropped_np = cropped.to_array()
        
        # 检查最小点数要求
        if cropped_np.shape[0] < FullCourtLocationConfig.MIN_SINGLE_FRAME_POINTS:
            return
        
        # 单帧预检查
        if not self._pre_check_single_frame(cropped):
            return
        
        # 将通过预检查的点云加入buffer
        self.buffer.append(cropped)
        rospy.loginfo(f"✅ 帧已添加到buffer，当前buffer长度: {len(self.buffer)}")

        # 处理累积的点云
        self.mag_header = header
        self._process_accumulated_clouds(x_range, y_range, z_range)

    def _get_crop_ranges(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """获取点云裁剪范围"""
        
        if self.use_auto_bbox and self.corners_cjkc is not None and len(self.corners_cjkc) == 4:
            # 使用自动边界框
            return self._calculate_auto_bbox()
        else:
            # 使用全场默认范围
            return (
                FullCourtLocationConfig.FULL_COURT_X_RANGE,
                FullCourtLocationConfig.FULL_COURT_Y_RANGE,
                FullCourtLocationConfig.FULL_COURT_Z_RANGE
            )

    def _calculate_auto_bbox(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """计算自动边界框"""
        
        right_top, left_top, left_bottom, right_bottom = self.corners_cjkc
        offset_x, offset_y, offset_z = 0.2, 0.2, 0.1
        
        # X方向取左右中点，向外扩展
        x_min = (left_bottom[0] + left_top[0]) / 2 - offset_x
        x_max = (right_bottom[0] + right_top[0]) / 2 + offset_x

        # Y方向取前后中点，向外扩展
        y_min = (left_bottom[1] + left_top[1]) / 2 - offset_y
        y_max = (right_bottom[1] + right_top[1]) / 2 + offset_y

        # Z方向取底部和顶部平均值±偏移
        z_min = (left_bottom[2] + right_bottom[2]) / 2 - offset_z
        z_max = (left_top[2] + right_top[2]) / 2 + offset_z

        return (x_min, x_max), (y_min, y_max), (z_min, z_max)

    # ======================== 点云处理函数 ========================
    
    def crop_box_filter(self, cloud: pcl.PointCloud, x_range: Tuple[float, float], 
                       y_range: Tuple[float, float], z_range: Tuple[float, float]) -> pcl.PointCloud:
        """点云立方体裁剪滤波"""
        
        # X轴滤波
        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("x")
        passthrough.set_filter_limits(x_range[0], x_range[1])
        cloud = passthrough.filter()

        # Y轴滤波
        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("y")
        passthrough.set_filter_limits(y_range[0], y_range[1])
        cloud = passthrough.filter()

        # Z轴滤波
        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("z")
        passthrough.set_filter_limits(z_range[0], z_range[1])
        cloud = passthrough.filter()

        return cloud

    def _process_accumulated_clouds(self, x_range: Tuple, y_range: Tuple, z_range: Tuple):
        """处理累积的点云数据"""
        
        try:
            # 合并所有缓冲区中的点云
            all_points_list = []
            for cloud_item in self.buffer:
                cloud_array = cloud_item.to_array()
                if len(cloud_array) > 0:
                    all_points_list.append(cloud_array)
            
            if not all_points_list:
                rospy.logwarn("Buffer中没有有效点云")
                return
                
            all_points = np.vstack(all_points_list)
            rospy.loginfo(f"合并后总点数: {all_points.shape[0]}")

            # 创建合并后的点云
            merged_cloud = pcl.PointCloud()
            merged_cloud.from_array(all_points)

            # 拟合篮框中心
            self._fit_basketball_center(merged_cloud, x_range, y_range, z_range)
            
        except Exception as e:
            rospy.logerr(f"点云合并或拟合过程出现异常: {e}")

    # ======================== 篮板检测和拟合 ========================
    
    def _fit_basketball_center(self, cloud: pcl.PointCloud, x_range: Tuple, y_range: Tuple, z_range: Tuple):
        """拟合篮框中心位置"""

        if cloud is None or cloud.size == 0:
            rospy.logwarn("输入点云为空，跳过处理")
            return

        points = np.asarray(cloud.to_array(), dtype=np.float32)
        
        if points.shape[0] == 0 or points.shape[1] < 3:
            rospy.logwarn("点云格式异常，转换后为空")
            return

        # 计算内外矩形
        inside_cloud, outer_points = self._extract_inner_outer_points(cloud, points)
        
        if outer_points is None or len(outer_points) == 0:
            rospy.logwarn("未找到有效的外部点云")
            return

        # 发布过滤后的点云用于可视化
        ros_cloud = pc2.create_cloud_xyz32(self.mag_header, outer_points)
        self.filter_cloud_pub.publish(ros_cloud)

        # 拟合四边形角点
        corners = self._fit_rectangle_from_pointcloud(outer_points)
        
        if not self._validate_corners(corners):
            return

        # 计算篮框中心
        basket_center = self._calculate_basket_center(corners)
        self.center = basket_center
        
        # 更新自动边界框设置
        self._update_auto_bbox_setting(corners)
        
        # 可视化
        self._publish_visualization_markers(corners, basket_center)

    def _extract_inner_outer_points(self, cloud: pcl.PointCloud, points: np.ndarray) -> Tuple[Optional[pcl.PointCloud], Optional[np.ndarray]]:
        """提取内部和外部点云"""
        
        x_vals, y_vals, z_vals = points[:, 0], points[:, 1], points[:, 2]

        # 使用分位数确定内矩形边界
        x_min, x_max = np.percentile(x_vals, [10, 90])
        y_min, y_max = np.percentile(y_vals, [10, 90])
        z_min, z_max = np.percentile(z_vals, [10, 90])

        if x_max - x_min < 1e-3 or z_max - z_min < 1e-3:
            rospy.logwarn("内矩形边界过小，可能异常")
            return None, None

        # 获取内部点云
        x_range_inner = (x_min, x_max)
        y_range_inner = (y_min, y_max)
        z_range_inner = (z_min, z_max)
        
        inside_cloud = self.crop_box_filter(cloud, x_range_inner, y_range_inner, z_range_inner)

        if inside_cloud is None or inside_cloud.size == 0:
            rospy.logwarn("裁剪后内部点云为空")
            return None, None

        # 计算外部点（不在内部的点）
        all_points = cloud.to_array()
        inside_points = inside_cloud.to_array()
        inside_set = set(map(tuple, inside_points))
        outer_points = np.array([pt for pt in all_points if tuple(pt) not in inside_set], dtype=np.float32)

        return inside_cloud, outer_points

    def _validate_corners(self, corners: List) -> bool:
        """验证角点是否有效"""
        
        if not corners or len(corners) != 4:
            rospy.logwarn("角点数量不为4，跳过该帧")
            return False

        corners_np = np.array(corners, dtype=np.float32)

        if np.allclose(corners_np, 0):
            rospy.logwarn("角点数据为零点，疑似拟合失败")
            return False
            
        return True

    def _calculate_basket_center(self, corners: List) -> Tuple[float, float, float]:
        """计算篮框中心位置"""
        
        corners_np = np.array(corners)
        right_top, left_top, left_bottom, right_bottom = corners_np
        
        # 计算法向量
        vec_ab = left_top - right_top    # 从右上到左上
        vec_ad = right_bottom - right_top  # 从右上到右下
        normal_vec = np.cross(vec_ab, vec_ad)
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        
        # 底边中点
        bottom_mid = (left_bottom + right_bottom) / 2.0
        
        # 前方偏移计算篮框位置
        offset_vec = normal_vec * FullCourtLocationConfig.HOOP_OFFSET_DISTANCE
        hoop_center = bottom_mid + offset_vec
        
        basket_center = tuple(hoop_center)
        rospy.loginfo(f"球框中心: {basket_center}")
        
        return basket_center

    def _update_auto_bbox_setting(self, corners: List):
        """更新自动边界框设置"""
        
        corners_np = np.array(corners)
        right_top, left_top, left_bottom, right_bottom = corners_np
        self.corners_cjkc = corners

        # 检查是否符合标准篮板尺寸
        ok, width, height = self._is_standard_backboard(right_top, left_top, left_bottom, right_bottom)
        
        if ok:
            rospy.loginfo(f"✅ 尺寸符合标准篮板：宽 {width:.3f} m，高 {height:.3f} m")
            self.use_auto_bbox = True
        else:
            self.use_auto_bbox = False

    # ======================== 几何计算工具函数 ========================
    
    def _fit_rectangle_from_pointcloud(self, points: np.ndarray) -> List:
        """从点云拟合矩形角点"""
        
        points = np.array(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            rospy.logwarn("点云数据格式不正确，shape=%s" % str(points.shape))
            return []
        
        # 使用XZ平面进行拟合
        xz_points = points[:, [0, 2]]
        origin_points = points

        # 过滤异常值
        xz_points_filtered = self._filter_percentile(xz_points)
        if len(xz_points_filtered) < 50:
            rospy.logwarn("点数量太少，无法拟合矩形")
            return []
        
        # 使用OpenCV拟合最小外接矩形
        rect = cv2.minAreaRect(xz_points_filtered)
        box = cv2.boxPoints(rect)
        box = self._reorder_box_points_2d(np.array(box))
        
        # 恢复3D坐标的Y值
        stable_y = self._get_stable_y_value(origin_points, rect, box)
        box_3d = [(x, stable_y, z) for x, z in box]
        
        return box_3d

    def _filter_percentile(self, points: np.ndarray, low: float = 0.05, high: float = 0.95) -> np.ndarray:
        """使用分位数过滤异常值"""
        
        x_min, x_max = np.percentile(points[:, 0], [low*100, high*100])
        z_min, z_max = np.percentile(points[:, 1], [(low+0.1)*100, (high-0.05)*100])
        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
               (points[:, 1] >= z_min) & (points[:, 1] <= z_max)
        return points[mask]

    def _get_stable_y_value(self, points_3d: np.ndarray, rect_2d, box_2d: np.ndarray) -> float:
        """获取稳定的Y坐标值"""
        
        center, (width, height), angle = rect_2d
        
        # 在矩形区域内查找点
        xz_coords = points_3d[:, [0, 2]]
        mask = self._point_in_rotated_rect(xz_coords, rect_2d)
        region_points = points_3d[mask]
        
        if len(region_points) >= 10:
            # 使用中位数，对异常值更鲁棒
            y_median = np.median(region_points[:, 1])
            y_std = np.std(region_points[:, 1])
            y_mean = np.mean(region_points[:, 1])
            
            if abs(y_median - y_mean) < y_std:
                return y_median
            else:
                # 过滤异常值后重新计算
                y_filtered = region_points[:, 1]
                mask_normal = np.abs(y_filtered - y_mean) <= 2 * y_std
                if np.sum(mask_normal) >= 5:
                    return np.median(y_filtered[mask_normal])
                else:
                    return y_median
        else:
            # 备选方案：使用距离加权的附近点
            tree = cKDTree(points_3d[:, [0, 2]])
            box_center = np.mean(box_2d, axis=0)
            distances, indices = tree.query(box_center, k=min(20, len(points_3d)))
            weights = 1.0 / (distances + 0.01)
            weights = weights / np.sum(weights)
            return np.average(points_3d[indices, 1], weights=weights)

    def _point_in_rotated_rect(self, points_xz: np.ndarray, rect_info) -> np.ndarray:
        """检查点是否在旋转矩形内"""
        
        center_pt, (w, h), ang = rect_info
        cos_a = np.cos(np.radians(ang))
        sin_a = np.sin(np.radians(ang))
        
        # 平移到矩形中心
        translated = points_xz - center_pt
        
        # 旋转到矩形坐标系
        rotated_x = translated[:, 0] * cos_a + translated[:, 1] * sin_a
        rotated_y = -translated[:, 0] * sin_a + translated[:, 1] * cos_a
        
        # 检查是否在矩形内
        margin = 0.1  # 10cm边界扩展
        return (np.abs(rotated_x) <= w/2 + margin) & (np.abs(rotated_y) <= h/2 + margin)

    def _reorder_box_points_2d(self, box_2d: np.ndarray) -> np.ndarray:
        """重新排列2D边界框角点，顺序：右上、左上、左下、右下"""
        
        points = np.array(box_2d)
        x_center = np.mean(points[:, 0])
        z_center = np.mean(points[:, 1])
        
        # 找到各象限的点
        indices = [None] * 4  # [右上, 左上, 左下, 右下]
        
        for i, point in enumerate(points):
            x, z = point
            if x > x_center and z > z_center:
                indices[0] = i  # 右上
            elif x <= x_center and z > z_center:
                indices[1] = i  # 左上
            elif x <= x_center and z <= z_center:
                indices[2] = i  # 左下
            elif x > x_center and z <= z_center:
                indices[3] = i  # 右下
        
        return points[indices]

    def _is_standard_backboard(self, right_top: np.ndarray, left_top: np.ndarray, 
                              left_bottom: np.ndarray, right_bottom: np.ndarray,
                              width_std: float = None, height_std: float = None, 
                              tolerance: float = None) -> Tuple[bool, float, float]:
        """检查是否为标准篮板尺寸"""
        
        if width_std is None:
            width_std = FullCourtLocationConfig.STANDARD_BACKBOARD_WIDTH
        if height_std is None:
            height_std = FullCourtLocationConfig.STANDARD_BACKBOARD_HEIGHT
        if tolerance is None:
            tolerance = FullCourtLocationConfig.BACKBOARD_TOLERANCE
        
        # 计算宽度（顶边和底边平均）
        width_top = np.linalg.norm(np.array(right_top) - np.array(left_top))
        width_bottom = np.linalg.norm(np.array(right_bottom) - np.array(left_bottom))
        width = (width_top + width_bottom) / 2

        # 计算高度（左边和右边平均）
        height_left = np.linalg.norm(np.array(left_top) - np.array(left_bottom))
        height_right = np.linalg.norm(np.array(right_top) - np.array(right_bottom))
        height = (height_left + height_right) / 2

        width_ok = abs(width - width_std) <= tolerance
        height_ok = abs(height - height_std) <= tolerance

        return width_ok and height_ok, width, height

    # ======================== 预检查函数 ========================
    
    def _pre_check_single_frame(self, cloud: pcl.PointCloud) -> bool:
        """对单帧点云进行快速预检查"""
        
        try:
            if cloud is None or cloud.size == 0:
                return False
            
            points = np.asarray(cloud.to_array(), dtype=np.float32)
            
            if points.shape[0] == 0 or points.shape[1] < 3:
                return False
            
            if points.shape[0] < FullCourtLocationConfig.MIN_PRECHECK_POINTS:
                return False
            
            # 快速内外矩形计算
            inside_cloud, outer_points = self._extract_inner_outer_points_relaxed(cloud, points)
            
            if outer_points is None or len(outer_points) < FullCourtLocationConfig.MIN_OUTER_POINTS:
                return False

            # 快速拟合检查
            corners = self._fit_rectangle_relaxed(outer_points)
            
            if not corners or len(corners) != 4:
                return False

            corners_np = np.array(corners, dtype=np.float32)
            if np.allclose(corners_np, 0):
                return False
            
            # 尺寸检查（使用更宽松的容差）
            right_top, left_top, left_bottom, right_bottom = corners_np
            ok, width, height = self._is_standard_backboard(
                right_top, left_top, left_bottom, right_bottom, 
                tolerance=0.3  # 放宽容差到50cm
            )
            
            if ok:
                rospy.loginfo(f"✅ 单帧预检查通过：宽 {width:.3f} m，高 {height:.3f} m")
                return True
            else:
                rospy.loginfo(f"❌ 单帧预检查失败：宽 {width:.3f} m，高 {height:.3f} m")
                return False
                
        except Exception as e:
            rospy.logwarn(f"单帧预检查出现异常: {e}")
            return False

    def _extract_inner_outer_points_relaxed(self, cloud: pcl.PointCloud, points: np.ndarray) -> Tuple[Optional[pcl.PointCloud], Optional[np.ndarray]]:
        """更宽松的内外点提取（用于预检查）"""
        
        x_vals, y_vals, z_vals = points[:, 0], points[:, 1], points[:, 2]

        # 使用更宽松的分位数范围
        x_min, x_max = np.percentile(x_vals, [20, 80])
        y_min, y_max = np.percentile(y_vals, [20, 80])
        z_min, z_max = np.percentile(z_vals, [20, 80])

        if x_max - x_min < 1e-3 or z_max - z_min < 1e-3:
            return None, None

        x_range_inner = (x_min, x_max)
        y_range_inner = (y_min, y_max)
        z_range_inner = (z_min, z_max)
        
        inside_cloud = self.crop_box_filter(cloud, x_range_inner, y_range_inner, z_range_inner)

        if inside_cloud is None or inside_cloud.size == 0:
            return None, None

        all_points = cloud.to_array()
        inside_points = inside_cloud.to_array()
        inside_set = set(map(tuple, inside_points))
        outer_points = np.array([pt for pt in all_points if tuple(pt) not in inside_set], dtype=np.float32)

        return inside_cloud, outer_points

    def _fit_rectangle_relaxed(self, points: np.ndarray) -> List:
        """更宽松的矩形拟合（用于预检查）"""
        
        points = np.array(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            return []

        xz_points = points[:, [0, 2]]
        
        # 更宽松的过滤
        xz_points_filtered = self._filter_percentile(xz_points, low=0.1, high=0.9)
        
        if len(xz_points_filtered) < FullCourtLocationConfig.MIN_FILTERED_POINTS:
            return []

        rect = cv2.minAreaRect(xz_points_filtered)
        box = cv2.boxPoints(rect)
        box = self._reorder_box_points_2d(np.array(box))

        # 简单的Y值恢复
        tree = cKDTree(points[:, [0, 2]])
        _, indices = tree.query(box, k=1)
        box_3d = [tuple(points[i]) for i in indices]

        return box_3d

    # ======================== 可视化函数 ========================
    
    def _publish_visualization_markers(self, corners: List, center: Tuple, marker_id: int = 0):
        """发布可视化标记"""
        
        # 矩形边界框
        marker_box = self._create_line_strip_marker(corners, marker_id, color=(0, 1, 0, 1))
        
        # 对角线
        marker_diag = self._create_diagonal_marker(corners, marker_id+1, color=(1, 0, 0, 1))
        
        # 中心点
        marker_center = self._create_sphere_marker(center, marker_id+2, color=(0, 0, 1, 1))

        # 发布标记
        self.marker_pub.publish(marker_box)
        self.marker_pub.publish(marker_diag)
        self.marker_pub.publish(marker_center)

    def _create_line_strip_marker(self, corners: List, marker_id: int, color: Tuple) -> Marker:
        """创建线条标记"""
        
        marker = Marker()
        marker.header = self.mag_header
        marker.ns = "rectangle"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.points = [Point(*c) for c in corners]
        marker.points.append(Point(*corners[0]))  # 闭合
        
        return marker

    def _create_diagonal_marker(self, corners: List, marker_id: int, color: Tuple) -> Marker:
        """创建对角线标记"""
        
        marker = Marker()
        marker.header = self.mag_header
        marker.ns = "rectangle"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        
        # 添加对角线
        marker.points.append(Point(*corners[0]))
        marker.points.append(Point(*corners[2]))
        marker.points.append(Point(*corners[1]))
        marker.points.append(Point(*corners[3]))
        
        return marker

    def _create_sphere_marker(self, center: Tuple, marker_id: int, color: Tuple) -> Marker:
        """创建球体标记"""
        
        marker = Marker()
        marker.header = self.mag_header
        marker.ns = "rectangle"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        
        return marker

    # ======================== 工具函数 ========================
    
    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        """归一化向量"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两向量间的夹角"""
        v1_u = FullCourtPointCloudProcessor.normalize_vector(v1)
        v2_u = FullCourtPointCloudProcessor.normalize_vector(v2)
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = np.arccos(dot)
        return angle


def main():
    """主函数"""
    try:
        rospy.init_node("full_court_location_node", anonymous=True)
        processor = FullCourtPointCloudProcessor()
        rospy.loginfo("🏀 全场篮球定位系统已启动")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logerr(f"系统异常: {e}")
    finally:
        rospy.loginfo("🛑 全场定位系统已停止")


if __name__ == '__main__':
    main()