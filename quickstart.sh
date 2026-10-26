#!/bin/bash

sleep 2
# 启动第一个终端，运行 livox_ros_driver2
gnome-terminal -- bash -c "roslaunch livox_ros_driver2 msg_MID360.launch; exec bash"
sleep 1

# 启动第二个终端，运行 fast_livo 映射
gnome-terminal -- bash -c "roslaunch fast_livo mapping_avia.launch; exec bash"
sleep 1

# gnome-terminal -- bash -c "rosrun esLocation esLocation.py  >> /home/g308/esLocation.log 2>&1; exec bash"
gnome-terminal -- bash -c "rosrun esLocation esLocation.py; exec bash"

sleep 1

gnome-terminal -- bash -c "timeout 90 rosbag record /tf /cloud_registered /aft_mapped_to_init; exec bash"
sleep 1

# 检查 rosrun 进程是否在运行
if pgrep -f "rosrun esLocation controller_data.py" > /dev/null; then
    echo "rosrun 进程正在运行，跳过后续命令"
    exit 1  # 退出码非0表示不执行后续操作
else
    echo "rosrun 进程未运行，继续执行命令..."
    gnome-terminal -- bash -c "rosrun esLocation controller_data.py ; exec bash"
    sleep 1
fi
