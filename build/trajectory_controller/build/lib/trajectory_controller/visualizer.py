#!/usr/bin/env python3
"""
Visualization Module
Author: Student
Date: 2025

This module provides visualization capabilities using matplotlib
for analyzing path smoothing, trajectory generation, and tracking performance.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import traceback


class VisualizerNode(Node):
    """
    ROS2 Node for visualization.
    
    Collects data and generates plots for analysis.
    """

    def __init__(self):
        """Initialize the visualizer node."""
        super().__init__('visualizer_node')

        # Data storage
        self.smooth_path = []
        self.trajectory = []
        self.robot_path = []
        self.tracking_errors = []
        self.timestamps = []

        # Subscribers
        self.smooth_path_sub = self.create_subscription(
            Path,
            '/smooth_path',
            self.smooth_path_callback,
            10
        )

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

        self.error_sub = self.create_subscription(
            Float64MultiArray,
            '/tracking_error',
            self.error_callback,
            10
        )

        # Timer for plotting
        self.plot_timer = self.create_timer(5.0, self.generate_plots)

        self.start_time = None
        self.get_logger().info('Visualizer Node initialized')

    def smooth_path_callback(self, msg: Path):
        """Store smooth path data."""
        try:
            self.smooth_path = [
                (pose.pose.position.x, pose.pose.position.y)
                for pose in msg.poses
            ]
        except Exception as e:
            self.get_logger().error(f'Error in smooth_path_callback: {str(e)}')

    def trajectory_callback(self, msg: Float64MultiArray):
        """Store trajectory data."""
        try:
            data = msg.data
            trajectory = []
            for i in range(0, len(data), 3):
                trajectory.append((data[i], data[i+1], data[i+2]))
            self.trajectory = trajectory
        except Exception as e:
            self.get_logger().error(f'Error in trajectory_callback: {str(e)}')

    def odom_callback(self, msg: Odometry):
        """Store robot position."""
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.robot_path.append((x, y))

            # Keep last 1000 points
            if len(self.robot_path) > 1000:
                self.robot_path.pop(0)

        except Exception as e:
            self.get_logger().error(f'Error in odom_callback: {str(e)}')

    def error_callback(self, msg: Float64MultiArray):
        """Store tracking error."""
        try:
            if self.start_time is None:
                self.start_time = self.get_clock().now().nanoseconds / 1e9

            current_time = self.get_clock().now().nanoseconds / 1e9
            elapsed = current_time - self.start_time

            error_data = msg.data
            if len(error_data) >= 3:
                self.timestamps.append(elapsed)
                self.tracking_errors.append(error_data[2])  # distance error

                # Keep last 1000 points
                if len(self.tracking_errors) > 1000:
                    self.timestamps.pop(0)
                    self.tracking_errors.pop(0)

        except Exception as e:
            self.get_logger().error(f'Error in error_callback: {str(e)}')

    def generate_plots(self):
        """Generate and save visualization plots."""
        try:
            if not self.smooth_path or not self.trajectory:
                self.get_logger().info('Waiting for data...')
                return

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            # Plot 1: Path and Trajectory
            ax1 = axes[0, 0]
            if self.smooth_path:
                smooth_x = [p[0] for p in self.smooth_path]
                smooth_y = [p[1] for p in self.smooth_path]
                ax1.plot(smooth_x, smooth_y, 'b-', linewidth=2, label='Smooth Path')

            if self.trajectory:
                traj_x = [t[0] for t in self.trajectory]
                traj_y = [t[1] for t in self.trajectory]
                ax1.plot(traj_x, traj_y, 'g--', linewidth=1.5, label='Trajectory')

            if self.robot_path:
                robot_x = [p[0] for p in self.robot_path]
                robot_y = [p[1] for p in self.robot_path]
                ax1.plot(robot_x, robot_y, 'r-', linewidth=1, label='Robot Path')

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('Path Tracking')
            ax1.legend()
            ax1.grid(True)
            ax1.axis('equal')

            # Plot 2: Tracking Error
            ax2 = axes[0, 1]
            if self.timestamps and self.tracking_errors:
                ax2.plot(self.timestamps, self.tracking_errors, 'r-', linewidth=2)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Tracking Error (m)')
                ax2.set_title('Tracking Error Over Time')
                ax2.grid(True)

                # Calculate statistics
                if len(self.tracking_errors) > 0:
                    mean_error = np.mean(self.tracking_errors)
                    max_error = np.max(self.tracking_errors)
                    ax2.axhline(y=mean_error, color='g', linestyle='--',
                                label=f'Mean: {mean_error:.3f}m')
                    ax2.legend()

            # Plot 3: Velocity Profile (from trajectory)
            ax3 = axes[1, 0]
            if self.trajectory and len(self.trajectory) > 1:
                times = [t[2] for t in self.trajectory]
                velocities = []

                for i in range(1, len(self.trajectory)):
                    dx = self.trajectory[i][0] - self.trajectory[i-1][0]
                    dy = self.trajectory[i][1] - self.trajectory[i-1][1]
                    dt = self.trajectory[i][2] - self.trajectory[i-1][2]

                    if dt > 1e-6:
                        velocity = np.sqrt(dx**2 + dy**2) / dt
                        velocities.append(velocity)
                    else:
                        velocities.append(0.0)

                ax3.plot(times[1:], velocities, 'b-', linewidth=2)
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Velocity (m/s)')
                ax3.set_title('Velocity Profile')
                ax3.grid(True)

            # Plot 4: Error Statistics
            ax4 = axes[1, 1]
            if len(self.tracking_errors) > 10:
                # Create histogram of errors
                ax4.hist(self.tracking_errors, bins=30, color='blue', alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Tracking Error (m)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Error Distribution')
                ax4.grid(True, alpha=0.3)

                # Add statistics text
                mean_error = np.mean(self.tracking_errors)
                std_error = np.std(self.tracking_errors)
                max_error = np.max(self.tracking_errors)
                stats_text = f'Mean: {mean_error:.3f}m\nStd: {std_error:.3f}m\nMax: {max_error:.3f}m'
                ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            # Save figure
            timestamp = self.get_clock().now().nanoseconds / 1e9
            filename = f'/tmp/trajectory_plot_{int(timestamp)}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

            self.get_logger().info(f'Saved plot to {filename}')

        except Exception as e:
            self.get_logger().error(
                f'Error generating plots: {str(e)}\n{traceback.format_exc()}'
            )


def main(args=None):
    """Main entry point for the visualizer node."""
    try:
        rclpy.init(args=args)
        node = VisualizerNode()
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
