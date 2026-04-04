# Instinct Onboard 环境配置完整流程


## Python安装

```bash
sudo apt-get install python3-venv
python3.10 -m venv --system-site-packages instinct_venv 
source instinct_venv/bin/activate
```
⚠️：不要安装过高版本python，会出现不同库兼容的冲突



## ROS2安装 

### 系统要求
测试过的系统和ros2版本
|系统|ros2 版本|
|--|--|
|Ubuntu 20.04|foxy|
|Ubuntu 22.04|humble (推荐)|
⚠️：以下指令针对humble版本 （官方参考网址：https://github.com/unitreerobotics/unitree_ros2/blob/master/README%20_zh.md）

### 安装指令

⚠️：尽量不要upgrade相关指令

```bash
git clone https://github.com/unitreerobotics/unitree_ros2

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt install curl gnupg2 lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
#将 ROS 2 软件源添加到系统列表
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-ros-base
sudo apt install ros-humble-rmw-cyclonedds-cpp
sudo apt install ros-humble-rosidl-generator-dds-idl
sudo apt install libyaml-cpp-dev

# 编译unitree_go和unitree_api功能包
sudo apt install python3-colcon-common-extensions
pip install empy==3.3.2 catkin_pkg lark setuptools pybind11
## 注意替换路径
cd ~/yixuan/instinct_onboard/unitree_ros2
source /opt/ros/humble/setup.bash
colcon build

# 配置网络
# 参考网址：https://github.com/unitreerobotics/unitree_ros2/blob/master/README%20_zh.md

#测试连接
source ./setup.sh              # 加载 ROS + 加载程序 + 配置网卡/DDS
source ./install/setup.bash    # 仅让系统认识新编译的程序
ros2 topic list
# 测试例程
./install/unitree_ros2_example/bin/read_low_state_hg
./install/unitree_ros2_example/bin/g1_dual_arm_example ./src/g1/low_level/g1_dual_arm_example/config
./install/unitree_ros2_example/bin/g1_ankle_swing_example ./src/g1/low_level/g1_ankle_swing_example/config
```

### 补充安装

如果已经存在ROS2，检查是否 install `unitree_hg` and `unitree_go` message definitions 

###  Make sure mcap storage for ros2 installed

- Make sure mcap storage for ros2 installed
    ```bash
    source /opt/ros/humble/setup.bash
    #注意替换路径
    source ~/yixuan/instinct_onboard/unitree_ros2/install/setup.bash
    source ~/yixuan/instinct_onboard/unitree_ros2/setup.sh
    source ./setup.sh

    sudo apt install ros-{ROS_VERSION}-rosbag2-storage-mcap
    sudo apt install ros-humble-rosbag2-storage-mcap
    ```




## crc安装

Follow the instruction of [crc_module](https://github.com/ZiwenZhuang/g1_crc) and copy the product (`crc_module.so`) to where you launch the python script.

## 使用专门为 CUDA 12 (JetPack 6) 提供的索引安装

pip install --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-gpu


- Install onboard python packages with automatic GPU detection (But with no OpenCV libraries)
    ```bash
    pip install -e .

    # 可以暂时只装 onnxruntime CPU版本，跳过 onnxruntime-gpu
    FORCE_CPU=1 pip install -e .

    ```
    This will automatically detect if GPU/CUDA is available and install the appropriate ONNX Runtime version.

    - Installation options:
        ```bash
        # Default installation (includes all dependencies including OpenCV libraries)
        pip install -e .[all]

        # No OpenCV dependencies installation
        pip install -e .[noopencv]
        ```

- Make sure `cv2` is accessible in the python environment. You can test it by running `import cv2` in the python shell.

    更合适的安装指令：pip install "opencv-python<4.9" "numpy==1.24.2"

    补充安装指令：
    pip install PyYAML
    source /opt/ros/humble/setup.bash && python3 -c "import rclpy; import yaml; from geometry_msgs.msg import TransformStamped; print('Success')"


## Policy observation

### 各 observation 参数的具体含义

- `joint_pos_ref`（未来参考关节位置，相对默认位）  
  来自参考动作（motion reference）未来若干帧的关节角，并减去默认站姿关节角（即 `joint_pos_ref - default_joint_pos`）。
- `joint_vel_ref`（未来参考关节速度）  
  来自参考动作未来若干帧的关节角速度目标
- `position_ref`（未来参考位移，motion-reference base frame）  
  把“当前时刻的参考根部位置”当作零点，计算未来每个参考帧相对这个零点移动了多少（x/y/z）；再用当前参考根部姿态做逆旋转，将该位移转换到当前 motion reference 的 base frame（不是机器人实时 base frame）。
- `rotation_ref`（未来参考旋转，tannorm）  
  未来参考根部姿态相对当前机器人姿态的旋转误差（tannorm 表达）。`tannorm` 把四元数转换成连续的 6 维向量表示（比欧拉角更平滑、对神经网络训练更稳定）。
- `depth_image`（深度图）  
- `projected_gravity`（重力投影向量）
  描述重力方向在机器人基坐标系下的投影分量。利用 IMU 实时获取的四元数，将世界坐标系下的单位重力向量 `[0, 0, -1]` 逆旋转至机体坐标系（本质为坐标系转换），从而得到反映机身倾斜状态的 3 维分量 `[gx, gy, gz]`。
- `base_ang_vel`  
  机体角速度（IMU获取）,包含 3 个数值（x/y/z）.
- `joint_pos`（相对默认位）  
  当前关节角减去默认站姿关节角，反映当前关节偏移。(init_state在env.yaml定义，是固定数值)
- `joint_vel`（相对默认速度）  
  当前关节角速度减去默认速度（默认设置为 0）
- `last_action`  
  上一个控制周期发给29个关节电机的指令，是相对于默认站姿的归一化位移量

### 每部分 size 的具体来源

当前模型配置关键常量：
- 机器人自由度：`NUM_JOINTS = 29`
- 参考帧数：`motion_reference.num_frames = 10`
- 深度图最终分辨率：`resize_shape = (18, 32)`
- 历史长度：`history_length = 8`（仅对部分 proprio 项有）

各项维度推导如下（先算原始 shape，再 flatten）：

- `joint_pos_ref` → `_get_joint_pos_ref_command_cmd_obs()`  
  shape = `(10, 29)`，size = `10 * 29 = 290`
- `joint_vel_ref` → `_get_joint_vel_ref_command_cmd_obs()`  
  shape = `(10, 29)`，size = `10 * 29 = 290`
- `position_ref` → `_get_position_b_ref_command_cmd_obs()`  
  shape = `(10, 3)`，size = `10 * 3 = 30`
- `rotation_ref` → `_get_rotation_ref_command_cmd_obs()`  
  shape = `(10, 6)`，size = `10 * 6 = 60`
- `depth_image` → `_get_visualizable_image_obs()`  
  shape = `(18, 32)`，size = `18 * 32 = 576`
- `projected_gravity` → `_get_projected_gravity_obs()` + history  
  单帧 `3` 维，history `8` 帧，size = `3 * 8 = 24`
- `base_ang_vel` → `_get_base_ang_vel_obs()` + history  
  单帧 `3` 维，history `8` 帧，size = `3 * 8 = 24`
- `joint_pos` → `_get_joint_pos_rel_obs()` + history  
  单帧 `29` 维，history `8` 帧，size = `29 * 8 = 232`
- `joint_vel` → `_get_joint_vel_rel_obs()` + history  
  单帧 `29` 维，history `8` 帧，size = `29 * 8 = 232`
- `last_action` → `_get_last_action_obs()` + history  
  单帧 `29` 维，history `8` 帧，size = `29 * 8 = 232`

完整 observation 总维度：

`290 + 290 + 30 + 60 + 576 + 24 + 24 + 232 + 232 + 232 = 1990`

推理时还会做一步“视觉编码后再拼接”：
- 先从 `1990` 维中切出 `depth_image` 的 `576` 维
- 经 depth encoder 输出 `32` 维 depth embedding
- actor 输入维度变为：`(1990 - 576) + 32 = 1446`


