#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from port import SerialPorter
import os

class Controller:
    def __init__(self):
        rospy.init_node('controller_data', anonymous=True)
        self.port = SerialPorter("/dev/serial_send")
        # 设置订阅者
        self.subscriber = rospy.Subscriber('/controller', Float32MultiArray, self.array_callback)
        
    def array_callback(self, msg):
        # 从消息中获取数组数据
        array_data = msg.data
        self.port.send_data(array_data)
        rospy.loginfo(f"发送了数据: {msg.data}")
        self.port.receive_data()
    def run(self):
        """运行节点"""
        rospy.spin()

if __name__ == '__main__':
    try:
        node = Controller()
        node.run()
    except rospy.ROSInterruptException:
        pass