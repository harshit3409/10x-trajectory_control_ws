# Trajectory Controller for Differential Drive Robots

A ROS2 package for path smoothing, trajectory generation, and trajectory tracking control for differential drive robots (TurtleBot3).

## Overview

This package implements a complete trajectory control system:

1. **Path Smoothing**: Converts discrete waypoints into smooth paths using cubic B-spline interpolation
2. **Trajectory Generation**: Creates time-parameterized trajectories with trapezoidal velocity profiles
3. **Trajectory Tracking**: PID controller for accurate trajectory following on differential drive robots

## Features

-  Modular, well-documented code following software engineering best practices
-  Comprehensive error handling and input validation
-  Unit tests for all major components
-  Real-time visualization with matplotlib
-  ROS2 Jazzy compatible
-  Gazebo simulation support

## System Requirements

- Ubuntu 24.04 LTS
- ROS2 Jazzy
- Python 3.12+
- TurtleBot3 packages

## Installation

### 1. Install ROS2 Jazzy (if not already installed)
```bash
# Add ROS2 repository
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2
sudo apt update
sudo apt upgrade
sudo apt install ros-jazzy-desktop
```

### 2. Install Dependencies
```bash
# Install ROS2 packages
sudo apt install -y python3-pip python3-colcon-common-extensions
sudo apt install -y ros-jazzy-gazebo-ros-pkgs
sudo apt install -y ros-jazzy-turtlebot3*
sudo apt install -y ros-jazzy-navigation2 ros-jazzy-nav2-bringup

# Install Python packages
pip3 install numpy scipy matplotlib

# Source ROS2
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Set TurtleBot3 model
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

### 3. Build the Package
```bash
# Create workspace
mkdir -p ~/trajectory_control_ws/src
cd ~/trajectory_control_ws/src

# Clone or copy the package
# (Assuming the package is already in src/trajectory_controller)

# Create resource directory
mkdir -p trajectory_controller/resource
touch trajectory_controller/resource/trajectory_controller

# Build
cd ~/trajectory_control_ws
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```

## Usage

### Running the Complete System

#### Terminal 1: Launch Gazebo Simulation
```bash
source ~/trajectory_control_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

#### Terminal 2: Launch Controller Nodes
```bash
source ~/trajectory_control_ws/install/setup.bash
ros2 launch trajectory_controller controller.launch.py
```

### Running Individual Nodes
```bash
# Path smoother only
ros2 run trajectory_controller path_smoother_node

# Trajectory generator only
ros2 run trajectory_controller trajectory_generator_node

# Trajectory tracker only
ros2 run trajectory_controller trajectory_tracker_node

# Visualizer only
ros2 run trajectory_controller visualizer_node
```

### Viewing Plots

Plots are automatically saved to `/tmp/trajectory_plot_*.png` every 5 seconds.
```bash
# View latest plot
eog /tmp/trajectory_plot_*.png
```

## Running Tests
```bash
cd ~/trajectory_control_ws

# Run all tests
colcon test --packages-select trajectory_controller

# View test results
colcon test-result --all --verbose

# Run individual test files
python3 src/trajectory_controller/test/test_path_smoother.py
python3 src/trajectory_controller/test/test_trajectory_generator.py
python3 src/trajectory_controller/test/test_trajectory_tracker.py
```

## Architecture

### Module Structure
````
trajectory_controller/
├── path_smoother.py          # Task 1: Path smoothing with B-splines
├── trajectory_generator.py   # Task 2: Time-parameterized trajectory generation
├── trajectory_tracker.py     # Task 3: PID-based trajectory tracking
└── visualizer.py             # Visualization and plotting

## AI Tools Used
1. **ChatGPT**- For Debugging
2. **Claude AI** - For syntax and commands
