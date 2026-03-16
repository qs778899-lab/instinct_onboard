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
