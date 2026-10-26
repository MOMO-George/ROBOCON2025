#!/bin/bash

# 设置日志文件
LOG_FILE="/tmp/competition_script.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== 脚本启动于 $(date) ====="

# 最大等待时间（秒）
MAX_WAIT=30
WAIT_STEP=1

# 等待设备出现
echo "等待 /dev/ttyUSB0 设备就绪..."
for ((i=0; i<=MAX_WAIT; i+=WAIT_STEP)); do
    if [ -e "/dev/ttyUSB0" ]; then
        echo "设备已检测到，等待时间: ${i}秒"
        break
    fi
    sleep $WAIT_STEP
    
    if [ $i -eq $MAX_WAIT ]; then
        echo "错误：等待超时（${MAX_WAIT}秒），/dev/ttyUSB0 设备未出现！"
        exit 1
    fi
done

# 设置设备权限
echo "尝试设置设备权限..."
for ((i=1; i<=3; i++)); do  # 最多尝试3次
    echo "308308" | sudo -S chmod 777 /dev/ttyUSB0 >/dev/null 2>&1
    
    if [ -w "/dev/ttyUSB0" ]; then
        echo "设备权限设置成功"
        break
    fi
    
    if [ $i -eq 3 ]; then
        echo "错误：/dev/ttyUSB0 权限设置失败！当前权限：$(ls -l /dev/ttyUSB0 | awk '{print $1}')"
        exit 1
    fi
    
    sleep 1  # 等待1秒后重试
done

# 检查并加载 ROS 环境
echo "检查 ROS 环境..."
if ! command -v roslaunch &> /dev/null; then
    echo "未检测到 roslaunch，尝试加载 ROS 环境..."
    source /opt/ros/noetic/setup.bash
    source /home/g308/mid360_ws/devel/setup.bash

    if ! command -v roslaunch &> /dev/null; then
        echo "错误：加载 ROS 环境失败，请确认 ROS 是否正确安装和配置"
        exit 1
    fi
fi

# 启动前延时
echo "延时 10 秒后启动 ROS 节点..."
sleep 10

roslaunch fast_livo competition.launch

echo "===== 脚本结束于 $(date) ====="
