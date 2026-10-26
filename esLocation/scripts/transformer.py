import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
import numpy as np


class WorldPoseTransformer:
    def __init__(self):
        # 参数设置
        self.self_offset = {"x": 0.0, "y": 0.3, "z": 0.0, "yaw_degree": -90.0}
        self.world_static_offset = {"x": 0.36, "y": 7.8, "yaw_degree": 0.0}
        self.keyIofoLocation = {"x": 0.0, "y": 0.0, "yaw_degree": 0.0}
        # 初始化发布器
        self.world_pose_pub = rospy.Publisher(
            "/robot_pose_world", Odometry, queue_size=10
        )

        # 初始化节点
        # rospy.init_node("dynamic_tf_node")
        rospy.loginfo("dynamic_tf_node started")

        # 订阅 odometry 消息
        rospy.Subscriber("/aft_mapped_to_init", Odometry, self.odometry_callback)

    def set_world_offset(self, offset_dict):
        """从字典设置 world_static_offset 的值"""
        required_keys = ["x", "y", "yaw_degree"]
        if not all(k in offset_dict for k in required_keys):
            rospy.logwarn(
                "world_static_offset 更新失败：缺少必要字段（需要包含 'x', 'y', 'yaw_degree'）"
            )
            return

        self.world_static_offset["x"] = offset_dict["x"]
        self.world_static_offset["y"] = offset_dict["y"]
        self.world_static_offset["yaw_degree"] = offset_dict["yaw_degree"]
        rospy.loginfo(f"world_static_offset updated: {self.world_static_offset}")

    def publish_pose(
        self, robot_x, robot_y, robot_z, robot_quat, stamp, frame_id="robot2world"
    ):
        pose_world_msg = Odometry()
        pose_world_msg.header.stamp = stamp
        pose_world_msg.header.frame_id = frame_id
        pose_world_msg.pose.pose.position.x = robot_x + self.world_static_offset["x"]
        pose_world_msg.pose.pose.position.y = robot_y + self.world_static_offset["y"]
        pose_world_msg.pose.pose.position.z = robot_z
        pose_world_msg.pose.pose.orientation.x = robot_quat[0]
        pose_world_msg.pose.pose.orientation.y = robot_quat[1]
        pose_world_msg.pose.pose.orientation.z = robot_quat[2]
        pose_world_msg.pose.pose.orientation.w = robot_quat[3]
        self.world_pose_pub.publish(pose_world_msg)

    def odometry_callback(self, msg: Odometry):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # 转换为欧拉角
        (roll, pitch, yaw) = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        # 坐标变换：local odom → base_link
        x_car = -position.y + self.self_offset["x"] * math.cos(math.pi / 2 + yaw)
        y_car = position.x + self.self_offset["y"] * math.sin(math.pi / 2 + yaw)
        x_self = x_car - self.self_offset["x"]
        y_self = y_car - self.self_offset["y"]

        # 添加 yaw 偏置
        robot2world_yaw = yaw + math.radians(self.world_static_offset["yaw_degree"])

        # 生成四元数
        robot2world_quat = quaternion_from_euler(roll, pitch, robot2world_yaw)

        self.keyIofoLocation = {"x": x_self, "y": y_self, "yaw_degree": robot2world_yaw}

        # 发布转换后的 pose
        self.publish_pose(x_self, y_self, 0, robot2world_quat, msg.header.stamp)

    def getKeyInfoLocation(self):
        return self.keyIofoLocation

    def start(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        transformer = WorldPoseTransformer()
        transformer.start()
    finally:
        rospy.loginfo("WorldPoseTransformer node exited.")
