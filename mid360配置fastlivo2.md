# 🚀 Livox MID360 配置教程（Livox-SDK2 + livox\_ros\_driver2）

## 一、前置依赖

确保系统安装以下依赖（Ubuntu 20.04 + ROS Noetic 示例）：

```bash
sudo apt update
sudo apt install git cmake build-essential
sudo apt install ros-noetic-desktop-full
```

---

## 二、创建工作空间

```bash
mkdir -p ~/livox_ws/src
cd ~/livox_ws/src
```

---

## 三、克隆 Livox-SDK2 和 livox\_ros\_driver2

```bash
# 克隆 SDK2
git clone https://github.com/Livox-SDK/Livox-SDK2.git

# 克隆 ROS 驱动
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
```

---

## 四、编译 Livox-SDK2

```bash
cd ~/livox_ws/src/Livox-SDK2
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

---

## 五、配置 Livox MID360 参数（重点）

编辑配置文件：

```bash
cd ~/livox_ws/src/livox_ros_driver2/config
cp livox_lidar_msg_config.json.example livox_lidar_msg_config.json
```

修改配置文件中两项：

1. **`host_net_info`** → 本机IP（与你的雷达IP处于同一网段，例如 192.168.1.100）
2. **`lidar_configs`** → 根据你的MID360广播码写入雷达的静态IP配置，例如：

广播码为 `47MDM540020550`，则 IP 为：`192.168.1.150`

```jsonc
{
  "lidar_type": "mid360",
  "host_net_info": {
    "host_ip": "192.168.1.100",
    "multicast_ip": "239.255.0.1",
    "data_port": 7601,
    "command_port": 7602
  },
  "lidar_configs": {
    "47MDM540020550": {
      "lidar_ip": "192.168.1.150",
      "cmd_port": 56100,
      "data_port": 56200
    }
  }
}
```

---

## 六、编译 livox\_ros\_driver2

```bash
cd ~/livox_ws
catkin_make
source devel/setup.bash
```

---

## 七、连接雷达

* 将你的电脑网口 IP 设置为 `192.168.1.100`（或上面配置中 `host_ip`）
* 雷达通过网线直连或交换机连接至电脑

---

## 八、运行驱动并可视化

### 启动驱动节点：

```bash
roslaunch livox_ros_driver2 livox_lidar_msg.launch
```

> 或者使用 RViz 可视化：

```bash
roslaunch livox_ros_driver2 rviz_MID360_launch.py
```

---

## 九、配置Fast-LIVO2


### 配置Sophus
```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build && cd build && cmake ..
make
sudo make install
```
可能遇到的问题
![](https://pic2.ziyuan.wang/user/jeason/2025/07/1751982902156_658af8fd67c83.png)

需要打开Sophus下sophus的so2.cpp文件
```cpp
 SO2::SO2()

{

   unit_complex_.real(1.);

   unit_complex_.imag(0.);

 }
```
改为
```cpp
SO2::SO2()
{
  //unit_complex_.real() = 1.;
  //unit_complex_.imag() = 0.;
  unit_complex_ = std::complex<double>(1,0);
}

```

### 配置Vikit
```bash
# Different from the one used in fast-livo1
cd catkin_ws/src
git clone https://github.com/xuankuzcr/rpg_vikit.git 
```

### build

```bash
cd ~/catkin_ws/src
git clone https://github.com/hku-mars/FAST-LIVO2
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```


### 启动测试
Download our collected rosbag files via OneDrive [Datasets](https://connecthkuhk-my.sharepoint.com/personal/zhengcr_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzhengcr%5Fconnect%5Fhku%5Fhk%2FDocuments%2Ffast%2Dlivo2%2Ddataset&ga=1).

```bash
roslaunch fast_livo mapping_avia.launch
rosbag play YOUR_DOWNLOADED.bag
```

## 融合mid360和fastlivo2

### 修改配置
进入`~/catkin_ws/src/FAST-LIVO2/config/avia.yaml`中    
修改示例如下:


```yaml
common:
  img_topic: "/left_camera/image"
  lid_topic: "/livox/lidar"
  imu_topic: "/livox/imu"
  img_en: 0
  lidar_en: 1
  ros_driver_bug_fix: false

extrin_calib:
  extrinsic_T: [0.04165, 0.02326, -0.0284]
  extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]
  Rcl: [0.00610193,-0.999863,-0.0154172,
        -0.00615449,0.0153796,-0.999863,
        0.999962,0.00619598,-0.0060598] 
  Pcl: [0.0194384, 0.104689,-0.0251952]

time_offset: 
  imu_time_offset: 0.0
  img_time_offset: 0.1
  exposure_time_init: 0.0

preprocess:
  point_filter_num: 1
  filter_size_surf: 0.1j
  lidar_type: 1 # Livox Avia LiDAR
  scan_line: 6
  blind: 0.8

vio:
  max_iterations: 5
  outlier_threshold: 1000 # 78 100 156 #100 200 500 700 infinite
  img_point_cov: 100 # 100 1000 
  patch_size: 8
  patch_pyrimid_level: 4
  normal_en: true
  raycast_en: false
  inverse_composition_en: false
  exposure_estimate_en: true
  inv_expo_cov: 0.1

imu:
  imu_en: true
  imu_int_frame: 30
  acc_cov: 0.5 # 0.2
  gyr_cov: 0.3 # 0.5
  b_acc_cov: 0.0001 # 0.1
  b_gyr_cov: 0.0001 # 0.1

lio:
  max_iterations: 5
  dept_err: 0.02
  beam_err: 0.05
  min_eigen_value: 0.0025 # 0.005
  voxel_size: 0.5
  max_layer: 2
  max_points_num: 50
  layer_init_num: [5, 5, 5, 5, 5]

local_map:  
  map_sliding_en: false
  half_map_size: 100
  sliding_thresh: 8

uav:
  imu_rate_odom: false
  gravity_align_en: false

publish:
  dense_map_en: true
  pub_effect_point_en: false 
  pub_plane_en: false
  pub_scan_num: 1
  blind_rgb_points: 0.0

evo:
  seq_name: "CBD_Building_01"
  pose_output_en: false

pcd_save:
  pcd_save_en: false
  colmap_output_en: false # need to set interval = -1
  filter_size_pcd: 0.15
  interval: -1
  # how many LiDAR frames saved in each pcd file;
  # -1 : all frames will be saved in ONE pcd file, may lead to memory crash when having too much frames.

```

