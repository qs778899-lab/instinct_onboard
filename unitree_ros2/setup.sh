#!/bin/bash
echo "Setup unitree ros2 environment (Humble)"

# 1. 修正 ROS 2 路径为 Humble
source /opt/ros/humble/setup.bash

# 2. 修正工作空间路径 (使用当前目录的绝对路径)
# source ./install/setup.bash
source /home/unitree/yixuan/instinct_onboard/unitree_ros2/install/setup.bash

# 3. 设置 DDS 实现
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 4. 关键：修正网卡名称
# 报错显示 enp3s0 不存在，请根据 ifconfig 的结果修改下面的 "lo" 或实际网卡名
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="enP8p1s0" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'

