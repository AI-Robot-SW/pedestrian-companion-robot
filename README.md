# OpenMind Interaction Autonomy

OpenMind-based AI-robot interaction and autonomy integration framework

## Overview

A framework that integrates the OpenMind platform with ROS2-based autonomous navigation systems.

- **OpenMind Platform**: Multimodal AI agent runtime with LLM/VLM integration
- **Vendor Autonomous Navigation**: ROS2-based autonomous navigation system (KIST collaboration)
- **Integration**: Communication between systems via DDS/Zenoh

## Features

### OpenMind Platform
- Multimodal AI agents
- LLM/VLM integration
- Hardware plugin support (ROS2, Zenoh, CycloneDDS)

### Vendor Autonomous Navigation
- RealSense camera-based perception
- GPU PointCloud generation
- BEV Occupancy Grid
- GPS-based Global Planner
- DWA Local Planner
- Unitree Go2 control

## Dependencies

### Hardware Requirements

#### Minimum Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space
- **Network**: Ethernet or WiFi connection

#### For OpenMind Platform
- **Camera**: USB webcam or compatible camera (for vision inputs)
- **Audio**: Microphone and speakers (for ASR/TTS)
- **Display**: Monitor for WebSim debugging interface

#### For Vendor Autonomous Navigation
- **GPU**: NVIDIA GPU with CUDA support (required for PointCloud and BEV processing)
  - NVIDIA Jetson AGX Orin / Thor (recommended)
  - NVIDIA GTX 1060 or better (desktop)
- **Camera**: Intel RealSense D435/D455 depth camera
- **Robot**: Unitree Go2 quadruped robot (for full autonomy)
- **GPS**: RTK-GPS module (for GPS-based navigation)
- **LIDAR**: RPLiDAR (optional, for SLAM)

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 22.04 (recommended) or Ubuntu 20.04
- **macOS**: macOS 12.0+ (limited support)
- **Windows**: Not officially supported

#### Core Software
- **Python**: >= 3.10
- **ROS2**: Humble Hawksbill or Iron Irwini
  - Required for vendor autonomous navigation
- **CUDA**: 11.0+ (for GPU acceleration)
- **CMake**: >= 3.5 (for ROS2 package building)

#### Package Managers
- **uv**: Python package manager
  ```bash
  # Install uv
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **colcon**: ROS2 build tool
  ```bash
  sudo apt install python3-colcon-common-extensions
  ```

#### System Libraries (Linux)
```bash
sudo apt-get update
sudo apt-get install -y \
    portaudio19-dev \
    python3-dev \
    ffmpeg \
    libasound2-dev \
    libv4l-dev \
    build-essential \
    cmake \
    python3-pip
```

#### Python Dependencies
Managed via `pyproject.toml`:
- Core: `numpy`, `opencv-python`, `pillow`, `json5`
- AI/ML: `torch`, `torchvision`, `tensorflow>=2.15.0`, `ultralytics`
- Communication: `eclipse-zenoh>=1.4.0`, `websockets`, `fastapi`
- Audio: `pyaudio`, `soundfile`, `sounddevice`
- Hardware: `pyserial`, `bleak`, `pynmeagps`
- Optional: `cyclonedds==0.10.2` (for DDS communication)

Install dependencies:
```bash
uv pip install -e .
# For DDS support
uv pip install -e ".[dds]"
```

#### ROS2 Dependencies
Vendor packages require:
- `rclpy`, `rclcpp` (ROS2 Python/C++ clients)
- `geometry_msgs`, `sensor_msgs`, `nav_msgs`
- `realsense2_camera` (RealSense ROS2 driver)
- `unitree_go` (Unitree Go2 messages)
- CUDA-enabled packages for GPU processing

## Quick Start

### OpenMind Platform

```bash
git clone <repository-url>
cd openmind-interaction-autonomy
uv venv
uv pip install -e .
uv run src/run.py <agent-name>
```

### Vendor ROS2

```bash
cd vendor/interaction_autonomous_navigation/Autonomous_Navigation
colcon build
source install/setup.bash

# RealSense camera
ros2 launch realsense2_camera rs_launch.py \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30 \
    enable_depth:=true enable_color:=true

# PointCloud generation
ros2 run pointcloud_xyzrgb pointcloud_gpu_node

# DWA Planner
ros2 run dwa_nav dwa_node
```

## Project Structure

```
.
├── src/              # OpenMind platform source
├── vendor/           # ROS2 autonomous navigation system
│   └── interaction_autonomous_navigation/
│       ├── Autonomous_Navigation/  # ROS2 workspace
│       └── LLMagent/              # LLM agent
├── config/           # Configuration files
└── pyproject.toml    # Python project configuration
```

## Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before making a pull request.

## License

MIT License
