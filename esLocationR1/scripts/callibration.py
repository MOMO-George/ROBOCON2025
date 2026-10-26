import rospy
from sensor_msgs.msg import LaserScan, Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
from sklearn.linear_model import RANSACRegressor
import math
import matplotlib.pyplot as plt
from collections import deque


class callibration:
    def __init__(self):
        self.targetAngle = {"back": 0, "left": 90, "front": 180, "right": -90}
        self.callibrationResult = None
        self.angleOffset = 51
        self.current_angle = 0
        self.is_calibration_active = False
        self.calibration_in_progress = False  # 标记是否正在进行校准计算
        self.pending_stop = False  # 标记是否有待处理的停止请求

        # 将数据话题订阅器设为None，稍后动态创建
        self.scan_sub = None
        self.imu_sub = None

        self.pub_Maker = rospy.Publisher("/RANSACline", Marker, queue_size=1)
        self.filterMin = 0.2
        self.final_lines = []

        # 点云叠加相关参数
        self.num_frames_to_stack = 10
        self.point_cloud_history = deque(maxlen=self.num_frames_to_stack)

        # 平滑处理参数
        self.alpha = 0.7
        self.previous_lines = None
        self.stability_threshold = 0.1

        print("校准系统初始化成功，等待外部信号...")

    def start_calibration(self):
        """开始校准，订阅数据话题"""
        if self.is_calibration_active:
            rospy.logwarn("校准已经在运行中")
            return

        rospy.loginfo("开始校准，订阅数据话题...")
        self.is_calibration_active = True
        self.calibration_in_progress = False
        self.pending_stop = False

        # 清空历史数据
        self.point_cloud_history.clear()
        self.previous_lines = None

        # 订阅数据话题
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.filter_laser_callback)
        self.imu_sub = rospy.Subscriber("/imu", Imu, self.imuCallback)

        rospy.loginfo("校准模式已激活")

    def request_stop_calibration(self):
        """请求停止校准（不会立即停止）"""
        if not self.is_calibration_active:
            rospy.logwarn("校准未在运行")
            return

        if self.calibration_in_progress:
            rospy.loginfo("校准正在进行中，将在完成后停止...")
            self.pending_stop = True
        else:
            self.stop_calibration()

    def stop_calibration(self):
        """实际停止校准"""
        rospy.loginfo("停止校准，取消订阅数据话题...")
        self.is_calibration_active = False
        self.calibration_in_progress = False
        self.pending_stop = False

        # 取消订阅
        if self.scan_sub is not None:
            self.scan_sub.unregister()
            self.scan_sub = None
        if self.imu_sub is not None:
            self.imu_sub.unregister()
            self.imu_sub = None

        rospy.loginfo("校准模式已停止")

    def imuCallback(self, msg: Imu):
        if not self.is_calibration_active:
            return
        ori = msg.orientation
        roll, pitch, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.current_angle = yaw

    def filter_laser_callback(self, msg: LaserScan):
        """
        过滤激光数据，叠加多帧点云，并拟合直线
        """
        if not self.is_calibration_active:
            return

        # 标记开始校准计算
        self.calibration_in_progress = True

        # 先保存原始数据用于后续参考角度计算
        original_angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        original_ranges = np.array(msg.ranges)

        # 用于直线拟合的过滤角度数据
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        # 添加角度过滤：只保留0度到150度的点云用于直线拟合
        angle_mask = (angles >= 0) & (angles <= np.deg2rad(140))
        angles = angles[angle_mask]
        ranges = ranges[angle_mask]

        # 过滤掉infinite和过近点
        valid_mask = np.isfinite(ranges) & (ranges > self.filterMin)
        angles = angles[valid_mask]
        ranges = ranges[valid_mask]

        # 转换笛卡尔坐标系
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T

        # 将当前帧的点云存入历史队列
        self.point_cloud_history.append(points)

        # 检查是否收集了足够的数据帧
        if len(self.point_cloud_history) < self.num_frames_to_stack:
            rospy.loginfo(
                f"正在收集数据帧用于叠加... {len(self.point_cloud_history)}/{self.num_frames_to_stack}"
            )
            self.calibration_in_progress = False

            # 检查是否有待处理的停止请求
            if self.pending_stop:
                self.stop_calibration()
            return

        # 将队列中的所有点云叠加成一个大的点云
        combined_points = np.vstack(list(self.point_cloud_history))

        # 基于叠加后的密集点云进行直线拟合
        current_lines = self.fit_two_orthogonal_lines(combined_points)

        if current_lines is not None:
            # 应用指数平滑处理，对结果进行二次平滑
            self.final_lines = self.smooth_lines(current_lines)

            if self.final_lines is not None:
                rospy.loginfo("在叠加后的点云上成功拟合了两条正交直线。")
                # 发布直线时，使用叠加后的点云范围来确定显示长度
                self.publish_lines(self.final_lines, combined_points)

                # 计算雷达到两条直线的垂直距离
                for idx, (slope, intercept, _) in enumerate(self.final_lines):
                    distance = abs(intercept) / math.sqrt(slope**2 + 1)
                    rospy.loginfo(
                        f"Line {idx}: 距离雷达中心的垂直距离为 {distance:.3f} 米"
                    )

                # 找出最靠近原点的那条直线（视为"后墙"）
                distances = []
                for slope, intercept, _ in self.final_lines:
                    d = abs(intercept) / math.sqrt(slope**2 + 1)
                    distances.append(d)

                rear_idx = np.argmin(distances)
                rear_line = self.final_lines[rear_idx]
                side_idx = 1 - rear_idx
                side_line = self.final_lines[side_idx]

                # 修正参考角度计算
                ref_angle_deg = self.targetAngle["back"] + self.angleOffset
                ref_angle_rad = np.deg2rad(ref_angle_deg)

                # 使用原始数据计算参考角度的激光距离
                ref_range = self.get_laser_range_at_angle(msg, ref_angle_rad)

                if ref_range is not None:
                    rospy.loginfo(
                        f"参考角度: {ref_angle_deg}°, 激光距离: {ref_range:.3f}m"
                    )
                    yaw_offset_deg, direction = self.compute_yaw_offset(
                        rear_line, ref_range
                    )
                    if yaw_offset_deg is not None:
                        rospy.loginfo(
                            f"估计yaw偏转角: {yaw_offset_deg:.2f}°，方向: {direction}"
                        )

                        y = distances[rear_idx]
                        x = distances[side_idx]
                        yaw_offset_deg = 0.0
                        self.callibrationResult = {
                            "x": x,
                            "y": y,
                            "yaw_degree": yaw_offset_deg,
                        }
                        rospy.loginfo(f"校准结果已保存: {self.callibrationResult}")
                else:
                    rospy.logwarn(f"无法获取参考角度 {ref_angle_deg}° 的激光距离")

        # 校准计算完成
        self.calibration_in_progress = False

        # 检查是否有待处理的停止请求
        if self.pending_stop:
            self.stop_calibration()

    def get_result(self):
        return self.callibrationResult

    def get_laser_range_at_angle(self, msg, target_angle_rad):
        """
        获取指定角度的激光距离
        """
        # 确保角度在有效范围内
        if target_angle_rad < msg.angle_min or target_angle_rad > msg.angle_max:
            rospy.logwarn(
                f"目标角度 {np.rad2deg(target_angle_rad):.1f}° 超出激光扫描范围 [{np.rad2deg(msg.angle_min):.1f}°, {np.rad2deg(msg.angle_max):.1f}°]"
            )
            return None

        # 计算索引
        angle_index = int((target_angle_rad - msg.angle_min) / msg.angle_increment)

        # 确保索引有效
        if 0 <= angle_index < len(msg.ranges):
            range_value = msg.ranges[angle_index]
            if np.isfinite(range_value) and range_value > self.filterMin:
                return range_value
            else:
                rospy.logwarn(
                    f"角度 {np.rad2deg(target_angle_rad):.1f}° 的激光距离无效: {range_value}"
                )
                return None
        else:
            rospy.logwarn(f"角度索引 {angle_index} 超出范围 [0, {len(msg.ranges) - 1}]")
            return None

    def smooth_lines(self, current_lines):
        """
        对拟合的直线进行指数加权平均平滑处理，减少抖动
        """
        if current_lines is None:
            return self.previous_lines

        # 指数加权平均
        if self.previous_lines is not None:
            smoothed_lines = []
            for i, (slope, intercept, angle) in enumerate(current_lines):
                # 确保 previous_lines 结构正确
                if i < len(self.previous_lines):
                    prev_slope, prev_intercept, prev_angle = self.previous_lines[i]

                    # 检查直线是否发生突变
                    slope_diff = abs(slope - prev_slope)
                    intercept_diff = abs(intercept - prev_intercept)

                    # 如果变化太大，可能是噪声，使用更大的平滑系数
                    alpha = (
                        0.3
                        if slope_diff > self.stability_threshold
                        or intercept_diff > self.stability_threshold
                        else self.alpha
                    )

                    # 指数加权平均
                    smooth_slope = alpha * slope + (1 - alpha) * prev_slope
                    smooth_intercept = alpha * intercept + (1 - alpha) * prev_intercept
                    smooth_angle = math.atan(smooth_slope)

                    smoothed_lines.append(
                        (smooth_slope, smooth_intercept, smooth_angle)
                    )
                else:
                    # 如果当前帧检测到的线比上一帧多，直接添加
                    smoothed_lines.append((slope, intercept, angle))

            self.previous_lines = smoothed_lines
            return smoothed_lines
        else:
            self.previous_lines = current_lines
            return current_lines

    def fit_two_orthogonal_lines(self, points, split_threshold=0.05, min_points=30):
        """
        使用 Split-and-Merge 算法从点云中提取线段，并筛选出两条接近正交的直线
        """

        def split(points):
            if len(points) < 2:
                return []

            start = points[0]
            end = points[-1]

            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                return []

            line_unit = line_vec / line_len
            distances = np.abs(np.cross(points - start, line_unit))  # 点到直线的距离
            max_dist_idx = np.argmax(distances)
            max_dist = distances[max_dist_idx]

            if max_dist > split_threshold:
                # 分裂
                left = split(points[: max_dist_idx + 1])
                right = split(points[max_dist_idx:])
                return left + right
            else:
                return [(start, end)]

        def fit_line(p1, p2):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0:
                slope = float("inf")
                intercept = p1[0]  # x = intercept
            else:
                slope = dy / dx
                intercept = p1[1] - slope * p1[0]
            angle = math.atan2(dy, dx)
            return slope, intercept, angle

        # 对点云按 x 坐标排序，以使路径连贯（防止乱序影响分割）
        sorted_points = points[np.argsort(points[:, 0])]

        # 执行 split
        segments = split(sorted_points)

        # 拟合线段
        fitted_lines = []
        for seg in segments:
            p1, p2 = seg
            if np.linalg.norm(p2 - p1) < 0.2 or len(points) < min_points:
                continue
            slope, intercept, angle = fit_line(p1, p2)
            fitted_lines.append((slope, intercept, angle))

        # 从所有拟合线中选择两条最接近正交的线段
        best_pair = None
        best_angle_diff = float("inf")
        for i in range(len(fitted_lines)):
            for j in range(i + 1, len(fitted_lines)):
                angle1 = fitted_lines[i][2]
                angle2 = fitted_lines[j][2]
                diff = abs(angle1 - angle2)
                diff = min(diff, np.pi - diff)
                if abs(diff - np.pi / 2) < best_angle_diff:
                    best_angle_diff = abs(diff - np.pi / 2)
                    best_pair = (fitted_lines[i], fitted_lines[j])

        if best_pair is not None and best_angle_diff < np.deg2rad(20):
            rospy.loginfo("Split-Merge算法成功找到正交线对")
            return best_pair
        else:
            rospy.logwarn("Split-Merge未能找到正交线对")
            return None

    def compute_yaw_offset(self, line, laser_distance):
        """
        根据一条直线和某个方向的激光测距，计算yaw偏转角及方向
        """
        slope, intercept, _ = line
        d = abs(intercept) / math.sqrt(slope**2 + 1)

        rospy.loginfo(f"直线到原点垂直距离: {d:.3f}m, 激光测距: {laser_distance:.3f}m")

        if laser_distance > d and laser_distance > self.filterMin:
            try:
                # 防止由于浮点数精度问题导致 acos 的参数略大于1
                ratio = min(1.0, d / laser_distance)
                delta_rad = math.acos(ratio)
                delta_deg = np.rad2deg(delta_rad)

                direction = "左" if slope > 0 else "右"
                return delta_deg, direction
            except ValueError as e:
                rospy.logwarn(
                    f"acos 计算错误: {e}, d={d}, laser_distance={laser_distance}"
                )
                return None, None
        else:
            # 当激光距离小于垂直距离时，可能意味着角度偏差很大或测量有误
            rospy.logwarn(
                f"激光距离({laser_distance:.3f}m)小于或接近于垂直距离({d:.3f}m)，无法精确计算偏角"
            )
            return 0.0, "未知"  # 在这种情况下可以认为偏差角为0

    def publish_lines(self, lines, points_for_range):
        for idx, (slope, intercept, _) in enumerate(lines):
            marker = Marker()
            marker.header.frame_id = "laser"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "fitted_lines"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.03

            if idx == 0:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            marker.color.a = 1.0

            p1 = Point()
            p2 = Point()

            x_min = float(points_for_range[:, 0].min())
            x_max = float(points_for_range[:, 0].max())

            p1.x = x_min
            p1.y = float(slope * p1.x + intercept)
            p1.z = 0.0

            p2.x = x_max
            p2.y = float(slope * p2.x + intercept)
            p2.z = 0.0

            marker.points.append(p1)
            marker.points.append(p2)

            self.pub_Maker.publish(marker)

    def get_calibration_status(self):
        """获取校准状态信息"""
        return {
            "is_active": self.is_calibration_active,
            "in_progress": self.calibration_in_progress,
            "pending_stop": self.pending_stop,
            "frames_collected": len(self.point_cloud_history),
            "frames_needed": self.num_frames_to_stack,
        }
