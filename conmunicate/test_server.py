from datetime import datetime
import time
import random
from server import UDPServer  # 确保与 server.py 同目录或已加入 PYTHONPATH

# 创建 UDPServer 实例（单例）
server = UDPServer()


# 定义自定义数据处理回调
def custom_data_handler(message):
    print("\n[自定义回调] 收到消息：")
    print(f"  发送方: {message.get('sender')}")
    print(f"  时间: {message.get('timestamp')}")
    print(f"  内容: {message.get('data')}")


# 注册回调处理函数
server.add_data_handler(custom_data_handler)

# 启动服务器
server.start()

# 循环发送 pose_data，每 2 秒一次
try:
    while True:
        pose_data = {
            "message_type": "pose",
            "timestamp": datetime.now().isoformat(),
            "client": "ExternalModule",
            "pose": {
                "position": {
                    "x": round(random.uniform(0, 10), 2),
                    "y": round(random.uniform(0, 10), 2),
                    "z": round(random.uniform(0, 10), 2),
                },
                "orientation": {
                    "x": round(random.uniform(-1, 1), 3),
                    "y": round(random.uniform(-1, 1), 3),
                    "z": round(random.uniform(-1, 1), 3),
                    "w": round(random.uniform(-1, 1), 3),
                },
            },
        }

        server.send_custom_data(pose_data)
        print("[发送] pose 数据已发送")
        time.sleep(2)

except KeyboardInterrupt:
    print("\n手动退出，停止服务器...")
    server.stop()
