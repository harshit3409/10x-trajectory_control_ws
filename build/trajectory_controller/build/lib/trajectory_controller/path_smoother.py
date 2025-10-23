#!/usr/bin/env python3
"""
Task 1: Path Smoothing Module
Author: Student
Date: 2025

This module implements path smoothing using cubic B-spline interpolation.
It converts discrete waypoints into a smooth, continuous path with C² continuity.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple
import traceback


class PathSmoother:
    """
    Path smoothing class using cubic B-spline interpolation.
    
    This class provides functionality to smooth discrete waypoints into
    a continuous path with guaranteed smoothness properties.
    """
    
    def __init__(self, smoothing_factor: float = 0.0, num_points: int = 100):
        """
        Initialize the path smoother.
        
        Args:
            smoothing_factor: Smoothing factor for spline (0 = interpolation, >0 = approximation)
            num_points: Number of points to sample along the smooth path
        """
        self.smoothing_factor = smoothing_factor
        self.num_points = num_points
        
    def smooth_path(self, waypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Smooth a list of 2D waypoints using cubic B-spline interpolation.
        
        Args:
            waypoints: List of (x, y) tuples representing waypoints
            
        Returns:
            List of (x, y) tuples representing smoothed path
            
        Raises:
            ValueError: If waypoints list has fewer than 3 points
            RuntimeError: If spline fitting fails
        """
        try:
            # Input validation
            if len(waypoints) < 3:
                raise ValueError(f"Need at least 3 waypoints for smoothing, got {len(waypoints)}")
            
            # Convert waypoints to numpy arrays
            waypoints_array = np.array(waypoints)
            x = waypoints_array[:, 0]
            y = waypoints_array[:, 1]
            
            # Check for duplicate consecutive points
            distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            if np.any(distances < 1e-6):
                # Remove duplicates
                unique_indices = [0]
                for i in range(1, len(waypoints)):
                    if distances[i-1] >= 1e-6:
                        unique_indices.append(i)
                x = x[unique_indices]
                y = y[unique_indices]
                
            # Determine spline degree (k) based on number of points
            k = min(3, len(x) - 1)  # Cubic if possible, else lower degree
            
            # Fit B-spline to waypoints
            # splprep returns tuple (tck, u) where tck contains spline parameters
            tck, u = splprep([x, y], s=self.smoothing_factor, k=k)
            
            # Sample the spline at regular intervals
            u_new = np.linspace(0, 1, self.num_points)
            smoothed = splev(u_new, tck)
            
            # Convert back to list of tuples
            smoothed_path = list(zip(smoothed[0], smoothed[1]))
            
            return smoothed_path
            
        except ValueError as e:
            raise ValueError(f"Path smoothing failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in path smoothing: {str(e)}\n{traceback.format_exc()}")
    
    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """
        Calculate the total length of a path.
        
        Args:
            path: List of (x, y) tuples
            
        Returns:
            Total path length in meters
        """
        try:
            if len(path) < 2:
                return 0.0
                
            path_array = np.array(path)
            differences = np.diff(path_array, axis=0)
            segment_lengths = np.sqrt(np.sum(differences**2, axis=1))
            return float(np.sum(segment_lengths))
            
        except Exception as e:
            raise RuntimeError(f"Error calculating path length: {str(e)}")
    
    def calculate_curvature(self, path: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate curvature at each point along the path.
        
        Args:
            path: List of (x, y) tuples
            
        Returns:
            List of curvature values (1/radius)
        """
        try:
            if len(path) < 3:
                return [0.0] * len(path)
                
            path_array = np.array(path)
            
            # Calculate first and second derivatives
            dx = np.gradient(path_array[:, 0])
            dy = np.gradient(path_array[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            # Curvature formula: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = np.abs(dx * ddy - dy * ddx)
            denominator = (dx**2 + dy**2)**(3/2)
            
            # Avoid division by zero
            denominator = np.where(denominator < 1e-10, 1e-10, denominator)
            curvature = numerator / denominator
            
            return curvature.tolist()
            
        except Exception as e:
            raise RuntimeError(f"Error calculating curvature: {str(e)}")


class PathSmootherNode(Node):
    """
    ROS2 Node for path smoothing.
    
    Subscribes to raw waypoints and publishes smoothed paths.
    """
    
    def __init__(self):
        """Initialize the path smoother node."""
        super().__init__('path_smoother_node')
        
        # Declare parameters
        self.declare_parameter('smoothing_factor', 0.0)
        self.declare_parameter('num_points', 100)
        self.declare_parameter('publish_rate', 1.0)
        
        # Get parameters
        smoothing_factor = self.get_parameter('smoothing_factor').value
        num_points = self.get_parameter('num_points').value
        publish_rate = self.get_parameter('publish_rate').value
        
        # Initialize smoother
        self.smoother = PathSmoother(smoothing_factor, num_points)
        
        # Publishers
        self.smooth_path_pub = self.create_publisher(Path, '/smooth_path', 10)
        
        # Define test waypoints (can be replaced with subscriber)
        self.waypoints = [
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 1.0),
            (3.0, 1.0),
            (4.0, 2.0),
            (5.0, 2.0),
            (6.0, 1.0),
            (7.0, 0.0),
        ]
        
        # Timer for publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_smooth_path)
        
        self.get_logger().info('Path Smoother Node initialized')
        self.get_logger().info(f'Smoothing {len(self.waypoints)} waypoints into {num_points} points')
    
    def publish_smooth_path(self):
        """Generate and publish smoothed path."""
        try:
            # Smooth the path
            smoothed_path = self.smoother.smooth_path(self.waypoints)
            
            # Calculate path metrics
            path_length = self.smoother.calculate_path_length(smoothed_path)
            curvatures = self.smoother.calculate_curvature(smoothed_path)
            max_curvature = max(curvatures) if curvatures else 0.0
            
            # Create Path message
            path_msg = Path()
            path_msg.header.frame_id = 'odom'
            path_msg.header.stamp = self.get_clock().now().to_msg()
            
            for x, y in smoothed_path:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            
            # Publish
            self.smooth_path_pub.publish(path_msg)
            
            self.get_logger().info(
                f'Published smooth path: {len(smoothed_path)} points, '
                f'Length: {path_length:.2f}m, Max curvature: {max_curvature:.3f}'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in path smoothing: {str(e)}\n{traceback.format_exc()}')


def main(args=None):
    """Main entry point for the path smoother node."""
    try:
        rclpy.init(args=args)
        node = PathSmootherNode()
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