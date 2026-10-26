#!/bin/bash

# 终止所有 roslaunch 进程的脚本
# 用法: ./kill_roslaunch.sh

echo "正在查找并终止 roslaunch 进程..."

# 1. 尝试优雅终止 (SIGTERM)
if pkill -f roslaunch; then
    echo "已发送终止信号到 roslaunch 进程"
else
    echo "未找到运行的 roslaunch 进程"
fi

# 2. 检查是否仍有残留进程
sleep 1  # 等待进程退出
if pgrep -f roslaunch > /dev/null; then
    echo "仍有 roslaunch 进程存活，尝试强制终止..."
    pkill -9 -f roslaunch  # 强制终止 (SIGKILL)
fi


# 1. 尝试优雅终止 (SIGTERM)
if pkill -f "rosrun esLocation esLocation.py"; then
    echo "已发送终止信号到 rosrun 进程"
else
    echo "未找到运行的 rosrun 进程"
fi
