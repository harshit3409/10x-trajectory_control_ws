#!/usr/bin/env python3
"""
Task 3: Trajectory Tracking Controller Module
Author: Student
Date: 2025

This module implements a PID controller for trajectory tracking on a
differential drive robot. It computes velocity commands to minimize
tracking errors.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
import numpy as np
from typing import List, Tuple
import traceback
import math


class PIDController:
    """
    PID Controller for trajectory tracking.
    
    Implements a standard PID controller with anti-windup and
    derivative filtering for smooth control.
    """
    
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = None):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limit: Maximum absolute output value (None for unlimited)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None
    
    def compute(self, error: float, current_time: float) -> float:
        """
        Compute control output.
        
        Args:
            error: Current error value
            current_time: Current timestamp
            
        Returns:
            Control output
        """
        try:
            # Calculate dt
            if self.previous_time is None:
                dt = 0.0
            else:
                dt = current_time - self.previous_time
            
            # Avoid division by zero
            if dt < 1e-6:
                dt = 1e-6
            
            # Proportional term
            p_term = self.kp * error
            
            # Integral term with anti-windup
            self.integral += error * dt
            # Clamp integral
            if self.output_limit is not None:
                max_integral = self.output_limit / (self.ki + 1e-6)
                self.integral = np.clip(self.integral, -max_integral, max_integral)
            i_term = self.ki * self.integral
            
            # Derivative term
            derivative = (error - self.previous_error) / dt
            d_term = self.kd * derivative
            
            # Calculate output
            output = p_term + i_term + d_term
            
            # Apply output limits
            if self.output_limit is not None:
                output = np.clip(output, -self.output_limit, self.output_limit)
            
            # Update state
            self.previous_error = error
            self.previous_time = current_time
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"Error in PID compute: {str(e)}")


class TrajectoryTracker:
    """
    Trajectory tracking controller for differential drive robots.
    
    Uses PID control for both linear and angular velocities to track
    a time-parameterized trajectory.
    """
    
    def __init__(
        self,
        linear_gains: Tuple[float, float, float] = (1.0, 0.0, 0.1),
        angular_gains: Tuple[float, float, float] = (2.0, 0.0, 0.2),
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 1.0,
        lookahead_distance: float = 0.3
    ):
        """
        Initialize the trajectory tracker.
        
        Args:
            linear_gains: (kp, ki, kd) for linear velocity PID
            angular_gains: (kp, ki, kd) for angular velocity PID
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
            lookahead_distance: Lookahead distance for path following (m)
        """
        # Initialize PID controllers
        self.linear_pid = PIDController(*linear_gains, max_linear_velocity)
        self.angular_pid = PIDController(*angular_gains, max_angular_velocity)
        
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.lookahead_distance = lookahead_distance
        
        # Trajectory storage
        self.trajectory = []
        self.start_time = None
    
    def set_trajectory(self, trajectory: List[Tuple[float, float, float]]):
        """
        Set a new trajectory to follow.
        
        Args:
            trajectory: List of (x, y, t) tuples
        """
        self.trajectory = trajectory
        self.start_time = None
        self.linear_pid.reset()
        self.angular_pid.reset()
    
    def get_target_pose(self, current_time: float) -> Tuple[float, float, float]:
        """
        Get the target pose at the current time.
        
        Args:
            current_time: Current time since trajectory start
            
        Returns:
            Tuple of (x, y, yaw) for target pose
        """
        try:
            if not self.trajectory:
                return (0.0, 0.0, 0.0)
            
            # Find the trajectory point at current time
            times = [t for _, _, t in self.trajectory]
            
            if current_time <= times[0]:
                x, y, _ = self.trajectory[0]
                return (x, y, 0.0)
            
            if current_time >= times[-1]:
                x, y, _ = self.trajectory[-1]
                return (x, y, 0.0)
            
            # Find interpolation indices
            idx = np.searchsorted(times, current_time)
            if idx >= len(self.trajectory):
                idx = len(self.trajectory) - 1
            
            # Linear interpolation
            t0, t1 = times[idx-1], times[idx]
            x0, y0, _ = self.trajectory[idx-1]
            x1, y1, _ = self.trajectory[idx]
            
            if abs(t1 - t0) < 1e-6:
                return (x1, y1, 0.0)
            
            alpha = (current_time - t0) / (t1 - t0)
            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)
            
            # Calculate target yaw
            yaw = math.atan2(y1 - y0, x1 - x0)
            
            return (x, y, yaw)
            
        except Exception as e:
            raise RuntimeError(f"Error getting target pose: {str(e)}")
    
    def compute_control(
        self,
        current_pose: Tuple[float, float, float],
        current_time: float
    ) -> Tuple[float, float]:
        """
        Compute control commands to track the trajectory.
        
        Args:
            current_pose: Current robot pose (x, y, yaw)
            current_time: Current time since trajectory start
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        try:
            if not self.trajectory:
                return (0.0, 0.0)
            
            # Initialize start time
            if self.start_time is None:
                self.start_time = current_time
            
            # Get elapsed time
            elapsed_time = current_time - self.start_time
            
            # Get target pose
            target_x, target_y, target_yaw = self.get_target_pose(elapsed_time)
            current_x, current_y, current_yaw = current_pose
            
            # Calculate errors in robot frame
            dx = target_x - current_x
            dy = target_y - current_y
            distance_error = math.sqrt(dx**2 + dy**2)
            
            # Calculate angle to target
            angle_to_target = math.atan2(dy, dx)
            
            # Calculate angular error (normalize to [-pi, pi])
            angular_error = self._normalize_angle(angle_to_target - current_yaw)
            
            # Compute control outputs using PID
            linear_velocity = self.linear_pid.compute(distance_error, current_time)
            angular_velocity = self.angular_pid.compute(angular_error, current_time)
            
            # Apply velocity limits
            linear_velocity = np.clip(
                linear_velocity,
                -self.max_linear_velocity,
                self.max_linear_velocity
            )
            angular_velocity = np.clip(
                angular_velocity,
                -self.max_angular_velocity,
                self.max_angular_velocity
            )
            
            # Reduce linear velocity when turning
            if abs(angular_error) > math.pi / 6:  # 30 degrees
                linear_velocity *= 0.5
            
            # Stop if at goal
            if elapsed_time >= self.trajectory[-1][2] and distance_error < 0.1:
                return (0.0, 0.0)
            
            return (linear_velocity, angular_velocity)
            
        except Exception as e:
            raise RuntimeError(f"Error computing control: {str(e)}")
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-pi, pi].
        
        Args:
            angle: Input angle in radians
            
        Returns:
            Normalized angle
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class TrajectoryTrackerNode(Node):
    """
    ROS2 Node for trajectory tracking.
    
    Subscribes to trajectory and odometry, publishes velocity commands.
    """
    
    def __init__(self):
        """Initialize the trajectory tracker node."""
        super().__init__('trajectory_tracker_node')
        
        # Declare parameters
        self.declare_parameter('linear_kp', 1.0)
        self.declare_parameter('linear_ki', 0.0)
        self.declare_parameter('linear_kd', 0.1)
        self.declare_parameter('angular_kp', 2.0)
        self.declare_parameter('angular_ki', 0.0)
        self.declare_parameter('angular_kd', 0.2)
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('control_rate', 20.0)
        
        # Get parameters
        linear_gains = (
            self.get_parameter('linear_kp').value,
            self.get_parameter('linear_ki').value,
            self.get_parameter('linear_kd').value
        )
        angular_gains = (
            self.get_parameter('angular_kp').value,
            self.get_parameter('angular_ki').value,
            self.get_parameter('angular_kd').value
        )
        max_linear_vel = self.get_parameter('max_linear_velocity').value
        max_angular_vel = self.get_parameter('max_angular_velocity').value
        control_rate = self.get_parameter('control_rate').value
        
        # Initialize tracker
        self.tracker = TrajectoryTracker(
            linear_gains,
            angular_gains,
            max_linear_vel,
            max_angular_vel
        )
        
        # Robot state
        self.current_pose = (0.0, 0.0, 0.0)
        self.trajectory_received = False
        
        # Subscribers
        self.trajectory_sub = self.create_subscription(
            Float64MultiArray,
            '/trajectory_data',
            self.trajectory_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.error_pub = self.create_publisher(Float64MultiArray, '/tracking_error', 10)
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / control_rate,
            self.control_loop
        )
        
        self.get_logger().info('Trajectory Tracker Node initialized')
        self.get_logger().info(
            f'Linear gains: {linear_gains}, Angular gains: {angular_gains}'
        )
    
    def trajectory_callback(self, msg: Float64MultiArray):
        """
        Callback for trajectory data messages.
        
        Args:
            msg: Float64MultiArray containing [x, y, t, x, y, t, ...]
        """
        try:
            # Parse trajectory data
            data = msg.data
            if len(data) % 3 != 0:
                self.get_logger().error('Invalid trajectory data length')
                return
            
            trajectory = []
            for i in range(0, len(data), 3):
                x, y, t = data[i], data[i+1], data[i+2]
                trajectory.append((x, y, t))
            
            # Set trajectory
            self.tracker.set_trajectory(trajectory)
            self.trajectory_received = True
            
            self.get_logger().info(f'Received trajectory with {len(trajectory)} points')
            
        except Exception as e:
            self.get_logger().error(
                f'Error in trajectory callback: {str(e)}\n{traceback.format_exc()}'
            )
    
    def odom_callback(self, msg: Odometry):
        """
        Callback for odometry messages.
        
        Args:
            msg: Odometry message
        """
        try:
            # Extract pose
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            
            # Extract yaw from quaternion
            orientation = msg.pose.pose.orientation
            siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
            cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            self.current_pose = (x, y, yaw)
            
        except Exception as e:
            self.get_logger().error(
                f'Error in odom callback: {str(e)}\n{traceback.format_exc()}'
            )
    
    def control_loop(self):
        """Main control loop executing at fixed rate."""
        try:
            if not self.trajectory_received:
                return
            
            # Get current time
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # Compute control
            linear_vel, angular_vel = self.tracker.compute_control(
                self.current_pose,
                current_time
            )
            
            # Create and publish command
            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            self.cmd_vel_pub.publish(cmd)
            
            # Calculate and publish error
            if self.tracker.trajectory:
                elapsed = current_time - (self.tracker.start_time or current_time)
                target_x, target_y, _ = self.tracker.get_target_pose(elapsed)
                error_x = target_x - self.current_pose[0]
                error_y = target_y - self.current_pose[1]
                error_distance = math.sqrt(error_x**2 + error_y**2)
                
                error_msg = Float64MultiArray()
                error_msg.data = [error_x, error_y, error_distance]
                self.error_pub.publish(error_msg)
            
        except Exception as e:
            self.get_logger().error(
                f'Error in control loop: {str(e)}\n{traceback.format_exc()}'
            )


def main(args=None):
    """Main entry point for the trajectory tracker node."""
    try:
        rclpy.init(args=args)
        node = TrajectoryTrackerNode()
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