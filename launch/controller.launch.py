#!/usr/bin/env python3
"""
Launch file for trajectory controller nodes
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for controller nodes."""
    
    # Get package directory
    pkg_dir = get_package_share_directory('trajectory_controller')
    
    # Parameters
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Path smoother node
    path_smoother_node = Node(
        package='trajectory_controller',
        executable='path_smoother_node',
        name='path_smoother',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time}
        ]
    )
    
    # Trajectory generator node
    trajectory_generator_node = Node(
        package='trajectory_controller',
        executable='trajectory_generator_node',
        name='trajectory_generator',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time}
        ]
    )
    
    # Trajectory tracker node
    trajectory_tracker_node = Node(
        package='trajectory_controller',
        executable='trajectory_tracker_node',
        name='trajectory_tracker',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time}
        ]
    )
    
    # Visualizer node
    visualizer_node = Node(
        package='trajectory_controller',
        executable='visualizer_node',
        name='visualizer',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        path_smoother_node,
        trajectory_generator_node,
        trajectory_tracker_node,
        visualizer_node
    ])