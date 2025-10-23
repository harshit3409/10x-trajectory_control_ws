#!/usr/bin/env python3
"""
Task 2: Trajectory Generation Module
Author: Student
Date: 2025

This module generates time-parameterized trajectories from smoothed paths
with velocity profiles (trapezoidal or constant velocity).
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
from typing import List, Tuple
import traceback


class TrajectoryGenerator:
    """
    Trajectory generation class with velocity profile support.
    
    Converts a smooth geometric path into a time-parameterized trajectory
    with specified velocity profiles for smooth motion.
    """
    
    def __init__(self, max_velocity: float = 0.5, max_acceleration: float = 0.3):
        """
        Initialize the trajectory generator.
        
        Args:
            max_velocity: Maximum linear velocity (m/s)
            max_acceleration: Maximum acceleration (m/s²)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
    
    def generate_trajectory(
        self, 
        path: List[Tuple[float, float]], 
        profile_type: str = 'trapezoidal'
    ) -> List[Tuple[float, float, float]]:
        """
        Generate time-parameterized trajectory from a smooth path.
        
        Args:
            path: List of (x, y) tuples representing the smooth path
            profile_type: Type of velocity profile ('trapezoidal' or 'constant')
            
        Returns:
            List of (x, y, t) tuples representing time-parameterized trajectory
            
        Raises:
            ValueError: If path has fewer than 2 points or invalid profile type
        """
        try:
            # Input validation
            if len(path) < 2:
                raise ValueError(f"Need at least 2 points for trajectory, got {len(path)}")
            
            if profile_type not in ['trapezoidal', 'constant']:
                raise ValueError(f"Invalid profile type: {profile_type}")
            
            # Calculate cumulative distance along path
            path_array = np.array(path)
            distances = self._calculate_cumulative_distance(path_array)
            total_distance = distances[-1]
            
            # Generate velocity profile
            if profile_type == 'trapezoidal':
                time_stamps = self._trapezoidal_profile(distances, total_distance)
            else:  # constant
                time_stamps = self._constant_profile(distances, total_distance)
            
            # Create time-parameterized trajectory
            trajectory = [
                (path[i][0], path[i][1], time_stamps[i])
                for i in range(len(path))
            ]
            
            return trajectory
            
        except ValueError as e:
            raise ValueError(f"Trajectory generation failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in trajectory generation: {str(e)}\n{traceback.format_exc()}")
    
    def _calculate_cumulative_distance(self, path_array: np.ndarray) -> np.ndarray:
        """
        Calculate cumulative distance along the path.
        
        Args:
            path_array: Nx2 array of path points
            
        Returns:
            Array of cumulative distances
        """
        differences = np.diff(path_array, axis=0)
        segment_lengths = np.sqrt(np.sum(differences**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
        return cumulative_distances
    
    def _trapezoidal_profile(
        self, 
        distances: np.ndarray, 
        total_distance: float
    ) -> np.ndarray:
        """
        Generate trapezoidal velocity profile.
        
        The profile has three phases:
        1. Acceleration: velocity increases from 0 to max
        2. Cruise: constant velocity at max
        3. Deceleration: velocity decreases from max to 0
        
        Args:
            distances: Cumulative distances along path
            total_distance: Total path length
            
        Returns:
            Array of time stamps corresponding to distances
        """
        try:
            # Calculate acceleration/deceleration distance
            accel_time = self.max_velocity / self.max_acceleration
            accel_distance = 0.5 * self.max_acceleration * accel_time**2
            
            # Check if we have enough distance for full trapezoid
            if 2 * accel_distance >= total_distance:
                # Triangular profile (no cruise phase)
                accel_distance = total_distance / 2
                accel_time = np.sqrt(2 * accel_distance / self.max_acceleration)
                cruise_distance = 0
                cruise_time = 0
                peak_velocity = self.max_acceleration * accel_time
            else:
                # Full trapezoidal profile
                cruise_distance = total_distance - 2 * accel_distance
                cruise_time = cruise_distance / self.max_velocity
                peak_velocity = self.max_velocity
            
            decel_distance = accel_distance
            decel_time = accel_time
            
            # Total time
            total_time = accel_time + cruise_time + decel_time
            
            # Generate time stamps for each distance
            time_stamps = np.zeros_like(distances)
            
            for i, d in enumerate(distances):
                if d <= accel_distance:
                    # Acceleration phase: d = 0.5 * a * t²
                    time_stamps[i] = np.sqrt(2 * d / self.max_acceleration)
                elif d <= accel_distance + cruise_distance:
                    # Cruise phase: d = d_accel + v_max * t
                    time_stamps[i] = accel_time + (d - accel_distance) / peak_velocity
                else:
                    # Deceleration phase
                    remaining_distance = total_distance - d
                    time_from_end = np.sqrt(2 * remaining_distance / self.max_acceleration)
                    time_stamps[i] = total_time - time_from_end
            
            return time_stamps
            
        except Exception as e:
            raise RuntimeError(f"Error in trapezoidal profile: {str(e)}")
    
    def _constant_profile(
        self, 
        distances: np.ndarray, 
        total_distance: float
    ) -> np.ndarray:
        """
        Generate constant velocity profile.
        
        Args:
            distances: Cumulative distances along path
            total_distance: Total path length
            
        Returns:
            Array of time stamps corresponding to distances
        """
        try:
            # Use half of max velocity for constant profile
            constant_velocity = self.max_velocity * 0.5
            time_stamps = distances / constant_velocity
            return time_stamps
            
        except Exception as e:
            raise RuntimeError(f"Error in constant profile: {str(e)}")
    
    def get_velocity_at_time(
        self, 
        trajectory: List[Tuple[float, float, float]], 
        time: float
    ) -> Tuple[float, float]:
        """
        Get velocity at a specific time along the trajectory.
        
        Args:
            trajectory: List of (x, y, t) tuples
            time: Query time
            
        Returns:
            Tuple of (velocity, angular_velocity)
        """
        try:
            if len(trajectory) < 2:
                return (0.0, 0.0)
            
            # Find surrounding points
            times = [t for _, _, t in trajectory]
            
            if time <= times[0]:
                return (0.0, 0.0)
            if time >= times[-1]:
                return (0.0, 0.0)
            
            # Find interpolation indices
            idx = np.searchsorted(times, time)
            if idx >= len(trajectory):
                idx = len(trajectory) - 1
            
            # Linear interpolation for velocity
            dt = times[idx] - times[idx-1]
            if dt < 1e-6:
                return (0.0, 0.0)
            
            dx = trajectory[idx][0] - trajectory[idx-1][0]
            dy = trajectory[idx][1] - trajectory[idx-1][1]
            
            velocity = np.sqrt(dx**2 + dy**2) / dt
            
            return (velocity, 0.0)  # Angular velocity computed by controller
            
        except Exception as e:
            raise RuntimeError(f"Error getting velocity: {str(e)}")


class TrajectoryGeneratorNode(Node):
    """
    ROS2 Node for trajectory generation.
    
    Subscribes to smoothed paths and publishes time-parameterized trajectories.
    """
    
    def __init__(self):
        """Initialize the trajectory generator node."""
        super().__init__('trajectory_generator_node')
        
        # Declare parameters
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('max_acceleration', 0.3)
        self.declare_parameter('profile_type', 'trapezoidal')
        
        # Get parameters
        max_velocity = self.get_parameter('max_velocity').value
        max_acceleration = self.get_parameter('max_acceleration').value
        self.profile_type = self.get_parameter('profile_type').value
        
        # Initialize generator
        self.generator = TrajectoryGenerator(max_velocity, max_acceleration)
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path,
            '/smooth_path',
            self.path_callback,
            10
        )
        
        # Publishers
        self.trajectory_pub = self.create_publisher(Path, '/trajectory', 10)
        self.trajectory_data_pub = self.create_publisher(
            Float64MultiArray, 
            '/trajectory_data', 
            10
        )
        
        self.get_logger().info('Trajectory Generator Node initialized')
        self.get_logger().info(f'Max velocity: {max_velocity} m/s, Profile: {self.profile_type}')
    
    def path_callback(self, msg: Path):
        """
        Callback for smooth path messages.
        
        Args:
            msg: Path message containing smoothed path
        """
        try:
            # Extract path points
            path = [(pose.pose.position.x, pose.pose.position.y) 
                   for pose in msg.poses]
            
            if len(path) < 2:
                self.get_logger().warn('Received path with less than 2 points')
                return
            
            # Generate trajectory
            trajectory = self.generator.generate_trajectory(path, self.profile_type)
            
            # Create trajectory message
            traj_msg = Path()
            traj_msg.header.frame_id = 'odom'
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Create data array for visualization
            data_array = Float64MultiArray()
            data_list = []
            
            for x, y, t in trajectory:
                pose = PoseStamped()
                pose.header = traj_msg.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                traj_msg.poses.append(pose)
                
                data_list.extend([x, y, t])
            
            data_array.data = data_list
            
            # Publish
            self.trajectory_pub.publish(traj_msg)
            self.trajectory_data_pub.publish(data_array)
            
            total_time = trajectory[-1][2] if trajectory else 0.0
            self.get_logger().info(
                f'Generated trajectory: {len(trajectory)} points, '
                f'Total time: {total_time:.2f}s'
            )
            
        except Exception as e:
            self.get_logger().error(
                f'Error in path callback: {str(e)}\n{traceback.format_exc()}'
            )


def main(args=None):
    """Main entry point for the trajectory generator node."""
    try:
        rclpy.init(args=args)
        node = TrajectoryGeneratorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in main: {str(e)}\n{traceback.format_exc()}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()