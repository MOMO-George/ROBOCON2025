import serial
import struct
import rospy
from sensor_msgs.msg import Imu
import math
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
import threading
from port import SerialPorter
from callibration import callibration
from transformer import WorldPoseTransformer

current_radian = 0
field = {"length": 7.5, "width": 8}  # 目前是半场

blankt = {"x": field["width"] / 2, "y": field["length"] - 0.905}  # 暂时直接用全局坐标


def radar2absolute(x_radar, y_radar, x_offset, y_offset, theta0):
    """
    雷达坐标系转换成绝对坐标系
    """
    cos_theta0 = np.cos(theta0)
    sin_theta0 = np.sin(theta0)
    x_abs = x_offset + x_radar * cos_theta0 - y_radar * cos_theta0
    y_abs = y_offset + x_radar * sin_theta0 + y_radar * cos_theta0
    return x_abs, y_abs


def main():
    rospy.init_node("calibration_main_node")
    rate = rospy.Rate(10)
    wpt = WorldPoseTransformer()
    # 初始化串口
    port = SerialPorter()
    if not port.open_port():
        rospy.logerr("串口打开失败，程序退出")
        return

    if not port.start_receiving():
        rospy.logerr("串口接收线程启动失败，程序退出")
        return

    # 初始化校准类
    calibration_system = callibration()

    # 校准状态标志
    calibration_active = False

    rospy.loginfo("主程序启动，等待下位机校准信号...")

    try:
        while not rospy.is_shutdown():
            # 检查串口是否接收到新的布尔值
            if port.has_bool_changed():
                current_bool = port.get_latest_bool()

                if current_bool and not calibration_active:
                    # 开始校准
                    rospy.loginfo("接收到开始校准信号")
                    calibration_system.start_calibration()
                    calibration_active = True
                    wpt.set_world_offset(callibration.get_result())
                elif not current_bool and calibration_active:
                    # 请求停止校准
                    rospy.loginfo("接收到停止校准信号")
                    calibration_system.request_stop_calibration()
                    calibration_active = False

            # 获取校准状态信息（可选，用于调试）
            status = calibration_system.get_calibration_status()
            if status["is_active"]:
                rospy.loginfo_throttle(
                    2,
                    f"校准状态: 活跃={status['is_active']}, "
                    f"进行中={status['in_progress']}, "
                    f"待停止={status['pending_stop']}, "
                    f"数据帧={status['frames_collected']}/{status['frames_needed']}",
                )
            loc = wpt.getKeyInfoLocation()
            absolute_x = loc["x"]
            absolute_y = loc["y"]
            distance_to_basket = math.sqrt(
                (blankt["x"] - absolute_x) ** 2 + (blankt["y"] - absolute_y) ** 2
            )

            target_yaw_angle = math.atan2(
                blankt["x"] - absolute_x, blankt["y"] - absolute_y
            )
            current_yaw = math.radians(loc["yaw_degree"])
            # 发送数据到下位机
            # port.send_data()
            print(f"[SLAM定位] 绝对坐标x: {absolute_x:.5f}, y: {absolute_y:.5f}")
            print(f"[SLAM定位] 当前角度: {math.degrees(current_yaw):.2f}°")
            print(f"[SLAM定位] 离篮筐距离: {distance_to_basket:.5f}")
            print(f"[SLAM定位] 目标角度: {math.degrees(target_yaw_angle):.2f}°")
            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("程序被ROS中断")
    except KeyboardInterrupt:
        rospy.loginfo("用户中断程序")
    finally:
        # 清理资源
        rospy.loginfo("正在清理资源...")
        if calibration_active:
            calibration_system.stop_calibration()
        port.close_port()
        rospy.loginfo("程序已安全退出")


if __name__ == "__main__":
    main()
