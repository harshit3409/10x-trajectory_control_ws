#!/usr/bin/env python3
"""
Task 1: Path Smoothing Module
Author: Student
Date: 2025

This module implements path smoothing using cubic B-spline interpolation.
It converts discrete waypoints into a smooth, continuous path with C² continuity.
Includes matplotlib visualization for immediate visual feedback.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


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


class PathSmootherVisualizer:
    """
    Visualization for path smoothing using matplotlib (non-interactive).
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.plot_counter = 0
        
    def generate_plot(self, waypoints, smoothed_path, curvatures, path_length, filename=None):
        """
        Generate and save a plot of the path smoothing results.
        
        Args:
            waypoints: Original waypoints
            smoothed_path: Smoothed path points
            curvatures: Curvature values
            path_length: Total path length
            filename: Output filename (optional)
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Path Smoothing Visualization', fontsize=16, fontweight='bold')
            
            # Plot 1: Path comparison
            ax1 = axes[0, 0]
            ax1.set_xlabel('X (m)', fontsize=10)
            ax1.set_ylabel('Y (m)', fontsize=10)
            ax1.set_title('Waypoints vs Smoothed Path', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')
            
            if waypoints:
                wp_x = [p[0] for p in waypoints]
                wp_y = [p[1] for p in waypoints]
                ax1.plot(wp_x, wp_y, 'ro-', markersize=8, linewidth=2, 
                        label=f'Waypoints ({len(waypoints)})', alpha=0.7)
            
            if smoothed_path:
                sp_x = [p[0] for p in smoothed_path]
                sp_y = [p[1] for p in smoothed_path]
                ax1.plot(sp_x, sp_y, 'b-', linewidth=2, 
                        label=f'Smoothed Path ({len(smoothed_path)})', alpha=0.8)
                
                # Mark start and end
                ax1.plot(sp_x[0], sp_y[0], 'go', markersize=12, 
                        label='Start', zorder=5)
                ax1.plot(sp_x[-1], sp_y[-1], 'rs', markersize=12, 
                        label='End', zorder=5)
            
            ax1.legend(loc='best', fontsize=9)
            
            # Plot 2: Curvature profile
            ax2 = axes[0, 1]
            ax2.set_xlabel('Point Index', fontsize=10)
            ax2.set_ylabel('Curvature (1/m)', fontsize=10)
            ax2.set_title('Path Curvature Profile', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            if curvatures:
                indices = np.arange(len(curvatures))
                ax2.plot(indices, curvatures, 'purple', linewidth=2)
                ax2.fill_between(indices, curvatures, alpha=0.3, color='purple')
                
                # Add mean line
                mean_curvature = np.mean(curvatures)
                ax2.axhline(y=mean_curvature, color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {mean_curvature:.4f}')
                ax2.legend(loc='best', fontsize=9)
            
            # Plot 3: Segment lengths
            ax3 = axes[1, 0]
            ax3.set_xlabel('Segment Index', fontsize=10)
            ax3.set_ylabel('Segment Length (m)', fontsize=10)
            ax3.set_title('Segment Length Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            if smoothed_path and len(smoothed_path) > 1:
                path_array = np.array(smoothed_path)
                differences = np.diff(path_array, axis=0)
                segment_lengths = np.sqrt(np.sum(differences**2, axis=1))
                
                indices = np.arange(len(segment_lengths))
                ax3.bar(indices, segment_lengths, color='teal', alpha=0.7, 
                       edgecolor='black', linewidth=0.5)
                
                # Add mean line
                mean_length = np.mean(segment_lengths)
                ax3.axhline(y=mean_length, color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {mean_length:.4f}m')
                ax3.legend(loc='best', fontsize=9)
            
            # Plot 4: Statistics table
            ax4 = axes[1, 1]
            ax4.axis('off')
            ax4.set_title('Path Statistics', fontsize=12, fontweight='bold')
            
            if waypoints and smoothed_path and curvatures:
                stats_text = self._generate_statistics_text(
                    waypoints, smoothed_path, curvatures, path_length
                )
                ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
            
            plt.tight_layout()
            
            # Save figure
            if filename is None:
                filename = f'/tmp/path_smoother_plot_{self.plot_counter}.png'
                self.plot_counter += 1
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to: {filename}")
            
            # Close figure to free memory
            plt.close(fig)
            
            return filename
            
        except Exception as e:
            print(f"Error generating plot: {str(e)}\n{traceback.format_exc()}")
            return None
    
    def _generate_statistics_text(self, waypoints, smoothed_path, curvatures, path_length):
        """Generate formatted statistics text."""
        max_curvature = max(curvatures) if curvatures else 0.0
        min_curvature = min(curvatures) if curvatures else 0.0
        mean_curvature = np.mean(curvatures) if curvatures else 0.0
        std_curvature = np.std(curvatures) if curvatures else 0.0
        
        # Calculate smoothness metric (lower is smoother)
        if len(curvatures) > 1:
            curvature_changes = np.abs(np.diff(curvatures))
            smoothness = np.mean(curvature_changes)
        else:
            smoothness = 0.0
        
        stats = f"""
╔═══════════════════════════════════════╗
║         PATH STATISTICS               ║
╠═══════════════════════════════════════╣
║ Original Waypoints:    {len(waypoints):>4d}          ║
║ Smoothed Points:       {len(smoothed_path):>4d}          ║
║                                       ║
║ Total Path Length:     {path_length:>6.2f} m       ║
║                                       ║
║ CURVATURE ANALYSIS:                   ║
║   Maximum:             {max_curvature:>6.4f} 1/m     ║
║   Minimum:             {min_curvature:>6.4f} 1/m     ║
║   Mean:                {mean_curvature:>6.4f} 1/m     ║
║   Std Dev:             {std_curvature:>6.4f} 1/m     ║
║                                       ║
║ Smoothness Index:      {smoothness:>6.4f}         ║
║   (lower = smoother)                  ║
╚═══════════════════════════════════════╝
        """
        return stats


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
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('save_plots', True)
        self.declare_parameter('plot_interval', 5)  # Save plot every N iterations
        
        # Get parameters
        smoothing_factor = self.get_parameter('smoothing_factor').value
        num_points = self.get_parameter('num_points').value
        publish_rate = self.get_parameter('publish_rate').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.save_plots = self.get_parameter('save_plots').value
        self.plot_interval = self.get_parameter('plot_interval').value
        
        # Initialize smoother
        self.smoother = PathSmoother(smoothing_factor, num_points)
        
        # Initialize visualizer
        if self.enable_visualization:
            self.visualizer = PathSmootherVisualizer()
        
        self.iteration_counter = 0
        
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
        if self.enable_visualization:
            self.get_logger().info(f'Visualization enabled - saving plots every {self.plot_interval} iterations')
    
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
            
            # Generate and save visualization plot
            if self.enable_visualization and self.save_plots:
                if self.iteration_counter % self.plot_interval == 0:
                    filename = self.visualizer.generate_plot(
                        self.waypoints,
                        smoothed_path,
                        curvatures,
                        path_length
                    )
                    if filename:
                        self.get_logger().info(f'Saved visualization to {filename}')
            
            self.iteration_counter += 1
            
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