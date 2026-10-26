import serial
import struct
import rospy
from sensor_msgs.msg import Imu
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
import os
import subprocess
from threading import Timer

current_radian = 0

class SerialPorter:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=2):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.buffer = bytearray()
        self.last_receive_time = time.time()
        self.receive_interval = 1.0  # 1秒轮询间隔
        self.receive_timer = None

    def start_receive_polling(self):
        """启动定时轮询接收"""
        self._receive_polling()

    def _receive_polling(self):
        """定时轮询接收函数"""
        if self.ser and self.ser.is_open:
            self.receive_data()
        
        # 设置下一次轮询
        self.receive_timer = Timer(self.receive_interval, self._receive_polling)
        self.receive_timer.daemon = True  # 设置为守护线程
        self.receive_timer.start()

    def receive_data(self):
        """接收数据并处理"""
        try:
            # 只读取当前可用的数据，不阻塞
            available = self.ser.in_waiting
            if available > 0:
                data = self.ser.read(available)
                self.buffer.extend(data)
                self.process_buffer()
        except Exception as e:
            print(f"接收数据异常: {e}")
            self.buffer.clear()

    def process_buffer(self):
        """处理缓冲区中的数据"""
        while True:
            # 查找帧头
            start = -1
            for i, byte in enumerate(self.buffer):
                if byte == 0x0A:  # 帧头
                    start = i
                    break
            
            if start == -1:
                self.buffer.clear()
                return
            
            # 查找帧尾(下一个0x0B)
            end = -1
            for i in range(start+1, len(self.buffer)):
                if self.buffer[i] == 0x0B:  # 帧尾
                    end = i
                    break
            
            if end == -1:
                # 保留未处理的头部数据
                if start > 0:
                    self.buffer = self.buffer[start:]
                return
            
            # 提取完整帧
            frame = self.buffer[start:end+1]
            self.buffer = self.buffer[end+1:]
            
            # 处理有效帧
            if len(frame) > 2:
                self.handle_frame(frame)

    def handle_frame(self, frame):
        """处理接收到的完整帧"""
        print(f"收到完整帧({len(frame)}字节): {frame.hex()}")
        # 异步执行脚本避免阻塞
        subprocess.Popen(["/bin/bash", "/home/g308/close.sh"])
        subprocess.Popen(["/bin/bash", "/home/g308/quickstart.sh"])

    def open_port(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.ser.reset_input_buffer()
            print(f"串口{self.port}成功打开")
            self.start_receive_polling()  # 启动轮询接收
            return True
        except Exception as exc:
            print("串口无法打开")
            return False

    def close_port(self):
        if self.receive_timer:
            self.receive_timer.cancel()
        if self.ser:
            self.ser.close()
            print(f"串口{self.port}成功关闭")

    def clear_buffer(self):
        if self.ser and self.ser.is_open:
            while self.ser.in_waiting > 0:
                self.ser.read(self.ser.in_waiting)

    def send_data(self, send_data):
        """发送数据，不受接收轮询影响"""
        try:
            if self.ser and self.ser.is_open:
                frame_head = bytes([0x0A])
                frame_tail = bytes([0x0B])
                data_body = b"".join(struct.pack("f", x) for x in send_data)
                full_data = frame_head + data_body + frame_tail
                self.ser.write(full_data)
                # print("发送成功")
            else:
                raise serial.SerialException
        except serial.SerialException as e:
            rospy.logerr(f"串口写入失败: {e}")
            self.reconnect_serial()

    def reconnect_serial(self):
        rospy.logwarn("尝试重新连接串口...")
        try:
            self.ser.close()
        except:
            pass
        try:
            # 添加重连逻辑
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            rospy.loginfo("串口重连成功")
            # 重启接收轮询
            if self.receive_timer:
                self.receive_timer.cancel()
            self.start_receive_polling()
        except Exception as e:
            rospy.logerr(f"重连失败: {e}")