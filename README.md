# Instinct Onboard

This is the onboard code for Project Instinct. Dsgiend for supporting the inference of the network on different robot manufacturing platforms.

***NOTE*** Current projects are only tested on Ubuntu 22.04 and ROS2 Humble with Unitree G1's Jetson Orin NX, 29Dof version.

## Prerequisites
- Ubuntu
- ROS2
- Python

### Installation (Unitree G1 Jetson Orin NX)

- JetPack
    ```bash
    sudo apt-get update
    sudo apt install nvidia-jetpack
    ```

- Install crc module

    Follow the instruction of [crc_module](https://github.com/ZiwenZhuang/g1_crc) and copy the product (`crc_module.so`) to where you launch the python script.

- Install `unitree_hg` and `unitree_go` message definitions (安装unitree_ros2后就无需再安装)


### Installation (Common)

- Make sure mcap storage for ros2 installed
    ```bash

    source /opt/ros/humble/setup.bash
    source ~/yixuan/instinct_onboard/unitree_ros2/install/setup.bash
    source ~/yixuan/instinct_onboard/unitree_ros2/setup.sh
    source ./setup.sh

    sudo apt install ros-{ROS_VERSION}-rosbag2-storage-mcap
    sudo apt install ros-humble-rosbag2-storage-mcap
    ```

- python virtual environment
    ```bash
    sudo apt-get install python3-venv
    python3 -m venv instinct_venv

    python3.10 -m venv instinct_venv #这个更高

    source instinct_venv/bin/activate
    ```
# . 使用专门为 CUDA 12 (JetPack 6) 提供的索引安装
pip install --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ onnxruntime-gpu

- Install onboard python packages with automatic GPU detection (But with no OpenCV libraries)
    ```bash
    pip install -e .

    # 暂时只装 装 onnxruntime，跳过 onnxruntime-gpu
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

    - `pip install opencv-python` or follow the instruction on [Geek for Geeks](https://www.geeksforgeeks.org/python/getting-started-with-opencv-cuda-module/) to build your own OpenCV with CUDA support.

- Notes:
    - ONNX Runtime version is auto-detected (GPU if available, CPU otherwise)
    - Use environment variables to override detection: `FORCE_CPU=1 pip install -e .` or `FORCE_GPU=1 pip install -e .`
    - If you want to build your GPU version OpenCV from source, you can install `instinct_onboard` with `[noopencv]` option.

    补充安装指令：
    pip install PyYAML
    source /opt/ros/humble/setup.bash && python3 -c "import rclpy; import yaml; from geometry_msgs.msg import TransformStamped; print('Success')"


## Code Structure Introduction

### ROS nodes

- In `instinct_onboard/ros_nodes/`, you can find the ROS nodes that are used to communicate with the robot.

- To avoid diamond inheritance, each function-specific ROS node should be implemented in a dedicated file with Mixin class.

- Please inherit everything you need in the script as well as the state machine logic in your main-entry script. (in `scripts/`)

### Agents

- In `instinct_onboard/agents/`, you can find the agents that are used to run the network (as well as collect the observations).

- Do NOT scale the action of the network output. The action scaling happens in the ros node side.
