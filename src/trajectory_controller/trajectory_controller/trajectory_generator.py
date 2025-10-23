#!/usr/bin/env python3
"""
Task 2: Trajectory Generation Module
Author: Student
Date: 2025

This module generates time-parameterized trajectories from smoothed paths
with velocity profiles (trapezoidal or constant velocity).
Includes matplotlib visualization for trajectory analysis.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
from typing import List, Tuple
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


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


class TrajectoryVisualizer:
    """
    Visualization for trajectory generation using matplotlib.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.plot_counter = 0
    
    def generate_plot(self, trajectory, profile_type, max_velocity, max_acceleration, filename=None):
        """
        Generate and save comprehensive trajectory visualization.
        
        Args:
            trajectory: List of (x, y, t) tuples
            profile_type: Type of velocity profile used
            max_velocity: Maximum velocity setting
            max_acceleration: Maximum acceleration setting
            filename: Output filename (optional)
        """
        try:
            if not trajectory or len(trajectory) < 2:
                print("Insufficient trajectory data for visualization")
                return None
            
            # Extract data
            x_coords = [point[0] for point in trajectory]
            y_coords = [point[1] for point in trajectory]
            time_stamps = [point[2] for point in trajectory]
            
            # Calculate velocities and accelerations
            velocities = self._calculate_velocities(trajectory)
            accelerations = self._calculate_accelerations(velocities, time_stamps)
            distances = self._calculate_distances(trajectory)
            
            # Create figure with 6 subplots
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            fig.suptitle(f'Trajectory Generation Analysis - {profile_type.upper()} Profile', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Trajectory in 2D space with time color coding
            ax1 = fig.add_subplot(gs[0, 0])
            scatter = ax1.scatter(x_coords, y_coords, c=time_stamps, cmap='viridis', 
                                 s=50, edgecolors='black', linewidth=0.5)
            ax1.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=2)
            ax1.plot(x_coords[0], y_coords[0], 'go', markersize=15, label='Start', zorder=5)
            ax1.plot(x_coords[-1], y_coords[-1], 'rs', markersize=15, label='End', zorder=5)
            ax1.set_xlabel('X Position (m)', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Y Position (m)', fontsize=10, fontweight='bold')
            ax1.set_title('Time-Parameterized Trajectory (x, y, t)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')
            ax1.legend(loc='best', fontsize=9)
            cbar1 = plt.colorbar(scatter, ax=ax1)
            cbar1.set_label('Time (s)', fontsize=9)
            
            # Plot 2: Velocity Profile over Time
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(time_stamps[:-1], velocities, 'b-', linewidth=2.5, label='Velocity')
            ax2.axhline(y=max_velocity, color='r', linestyle='--', linewidth=2, 
                       label=f'Max Velocity: {max_velocity:.2f} m/s')
            ax2.fill_between(time_stamps[:-1], velocities, alpha=0.3, color='blue')
            ax2.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Velocity (m/s)', fontsize=10, fontweight='bold')
            ax2.set_title('Velocity Profile vs Time', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=9)
            
            # Plot 3: Acceleration Profile over Time
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(time_stamps[:-2], accelerations, 'r-', linewidth=2.5, label='Acceleration')
            ax3.axhline(y=max_acceleration, color='orange', linestyle='--', linewidth=2, 
                       label=f'Max Accel: {max_acceleration:.2f} m/s²')
            ax3.axhline(y=-max_acceleration, color='orange', linestyle='--', linewidth=2, 
                       label=f'Max Decel: -{max_acceleration:.2f} m/s²')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
            ax3.fill_between(time_stamps[:-2], accelerations, alpha=0.3, color='red')
            ax3.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Acceleration (m/s²)', fontsize=10, fontweight='bold')
            ax3.set_title('Acceleration Profile vs Time', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='best', fontsize=9)
            
            # Plot 4: Velocity vs Distance
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(distances[:-1], velocities, 'g-', linewidth=2.5, marker='o', 
                    markersize=4, alpha=0.7)
            ax4.set_xlabel('Distance along path (m)', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Velocity (m/s)', fontsize=10, fontweight='bold')
            ax4.set_title('Velocity Profile vs Distance', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: 3D Trajectory (X, Y, Time)
            ax5 = fig.add_subplot(gs[2, 0], projection='3d')
            ax5.plot(x_coords, y_coords, time_stamps, 'b-', linewidth=2, alpha=0.7)
            ax5.scatter(x_coords, y_coords, time_stamps, c=time_stamps, cmap='viridis', 
                       s=30, edgecolors='black', linewidth=0.5)
            ax5.scatter([x_coords[0]], [y_coords[0]], [time_stamps[0]], 
                       color='green', s=100, marker='o', label='Start')
            ax5.scatter([x_coords[-1]], [y_coords[-1]], [time_stamps[-1]], 
                       color='red', s=100, marker='s', label='End')
            ax5.set_xlabel('X (m)', fontsize=9, fontweight='bold')
            ax5.set_ylabel('Y (m)', fontsize=9, fontweight='bold')
            ax5.set_zlabel('Time (s)', fontsize=9, fontweight='bold')
            ax5.set_title('3D Trajectory (x, y, t)', fontsize=12, fontweight='bold')
            ax5.legend(loc='best', fontsize=8)
            
            # Plot 6: Statistics and Trajectory Data Table
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.axis('off')
            ax6.set_title('Trajectory Statistics & Sample Data', fontsize=12, fontweight='bold')
            
            # Generate statistics text
            stats_text = self._generate_statistics_text(
                trajectory, velocities, accelerations, distances, 
                profile_type, max_velocity, max_acceleration
            )
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
            
            # Save figure
            if filename is None:
                filename = f'/tmp/trajectory_generator_plot_{self.plot_counter}.png'
                self.plot_counter += 1
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Trajectory plot saved to: {filename}")
            
            # Also save trajectory data as text file
            data_filename = filename.replace('.png', '_data.txt')
            self._save_trajectory_data(trajectory, data_filename)
            
            plt.close(fig)
            return filename
            
        except Exception as e:
            print(f"Error generating trajectory plot: {str(e)}\n{traceback.format_exc()}")
            return None
    
    def _calculate_velocities(self, trajectory):
        """Calculate velocities between trajectory points."""
        velocities = []
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            dt = trajectory[i+1][2] - trajectory[i][2]
            
            if dt > 1e-6:
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
            else:
                velocities.append(0.0)
        
        return velocities
    
    def _calculate_accelerations(self, velocities, time_stamps):
        """Calculate accelerations from velocities."""
        accelerations = []
        for i in range(len(velocities) - 1):
            dv = velocities[i+1] - velocities[i]
            dt = time_stamps[i+1] - time_stamps[i]
            
            if dt > 1e-6:
                acceleration = dv / dt
                accelerations.append(acceleration)
            else:
                accelerations.append(0.0)
        
        return accelerations
    
    def _calculate_distances(self, trajectory):
        """Calculate cumulative distances along trajectory."""
        distances = [0.0]
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distances[-1] + distance)
        
        return distances
    
    def _generate_statistics_text(self, trajectory, velocities, accelerations, 
                                  distances, profile_type, max_velocity, max_acceleration):
        """Generate formatted statistics text."""
        total_time = trajectory[-1][2] - trajectory[0][2]
        total_distance = distances[-1]
        avg_velocity = np.mean(velocities) if velocities else 0.0
        max_vel_actual = np.max(velocities) if velocities else 0.0
        max_accel_actual = np.max(np.abs(accelerations)) if accelerations else 0.0
        
        # Sample trajectory points (first 5 and last 5)
        sample_size = min(5, len(trajectory))
        
        stats = f"""
╔════════════════════════════════════════════════╗
║         TRAJECTORY STATISTICS                  ║
╠════════════════════════════════════════════════╣
║ Profile Type:        {profile_type.upper():<20s}     ║
║ Number of Points:    {len(trajectory):<4d}                       ║
║ Total Distance:      {total_distance:>6.2f} m                  ║
║ Total Time:          {total_time:>6.2f} s                  ║
║ Avg Velocity:        {avg_velocity:>6.3f} m/s                ║
║ Max Velocity (set):  {max_velocity:>6.3f} m/s                ║
║ Max Velocity (act):  {max_vel_actual:>6.3f} m/s                ║
║ Max Accel (set):     {max_acceleration:>6.3f} m/s²               ║
║ Max Accel (act):     {max_accel_actual:>6.3f} m/s²               ║
╠════════════════════════════════════════════════╣
║ SAMPLE TRAJECTORY POINTS (x, y, t):           ║
║ Format: [(x0, y0, t0), ..., (xn, yn, tn)]     ║
╠════════════════════════════════════════════════╣
"""
        # Add first few points
        for i in range(sample_size):
            x, y, t = trajectory[i]
            stats += f"║ [{i:3d}] ({x:>6.2f}, {y:>6.2f}, {t:>6.2f})                ║\n"
        
        if len(trajectory) > sample_size * 2:
            stats += f"║ ...                                            ║\n"
        
        # Add last few points
        for i in range(max(sample_size, len(trajectory) - sample_size), len(trajectory)):
            x, y, t = trajectory[i]
            stats += f"║ [{i:3d}] ({x:>6.2f}, {y:>6.2f}, {t:>6.2f})                ║\n"
        
        stats += "╚════════════════════════════════════════════════╝"
        
        return stats
    
    def _save_trajectory_data(self, trajectory, filename):
        """Save trajectory data to a text file."""
        try:
            with open(filename, 'w') as f:
                f.write("# Time-Parameterized Trajectory Data\n")
                f.write("# Format: trajectory = [(x0, y0, t0), (x1, y1, t1), ..., (xn, yn, tn)]\n")
                f.write("#\n")
                f.write(f"# Total points: {len(trajectory)}\n")
                f.write("#\n")
                f.write("# Index, X (m), Y (m), Time (s)\n")
                f.write("# " + "-"*50 + "\n")
                
                for i, (x, y, t) in enumerate(trajectory):
                    f.write(f"{i:4d}, {x:8.4f}, {y:8.4f}, {t:8.4f}\n")
                
                f.write("\n# Python format:\n")
                f.write("trajectory = [\n")
                for i, (x, y, t) in enumerate(trajectory):
                    if i < len(trajectory) - 1:
                        f.write(f"    ({x:.4f}, {y:.4f}, {t:.4f}),\n")
                    else:
                        f.write(f"    ({x:.4f}, {y:.4f}, {t:.4f})\n")
                f.write("]\n")
            
            print(f"✓ Trajectory data saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving trajectory data: {str(e)}")


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
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('save_plots', True)
        self.declare_parameter('plot_interval', 5)
        
        # Get parameters
        max_velocity = self.get_parameter('max_velocity').value
        max_acceleration = self.get_parameter('max_acceleration').value
        self.profile_type = self.get_parameter('profile_type').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.save_plots = self.get_parameter('save_plots').value
        self.plot_interval = self.get_parameter('plot_interval').value
        
        # Initialize generator
        self.generator = TrajectoryGenerator(max_velocity, max_acceleration)
        
        # Initialize visualizer
        if self.enable_visualization:
            self.visualizer = TrajectoryVisualizer()
        
        self.iteration_counter = 0
        
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
        if self.enable_visualization:
            self.get_logger().info(f'Visualization enabled - saving plots every {self.plot_interval} iterations')
    
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
            
            # Log sample trajectory points in the required format
            self.get_logger().info("=" * 60)
            self.get_logger().info("TIME-PARAMETERIZED TRAJECTORY")
            self.get_logger().info(f"Format: trajectory = [(x0, y0, t0), (x1, y1, t1), ..., (xn, yn, tn)]")
            self.get_logger().info("=" * 60)
            
            # Print first few and last few points
            sample_size = min(5, len(trajectory))
            for i in range(sample_size):
                x, y, t = trajectory[i]
                self.get_logger().info(f"  [{i:3d}] ({x:7.3f}, {y:7.3f}, {t:7.3f})")
            
            if len(trajectory) > sample_size * 2:
                self.get_logger().info("  ...")
            
            for i in range(max(sample_size, len(trajectory) - sample_size), len(trajectory)):
                x, y, t = trajectory[i]
                self.get_logger().info(f"  [{i:3d}] ({x:7.3f}, {y:7.3f}, {t:7.3f})")
            
            self.get_logger().info("=" * 60)
            
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
            
            # Generate and save visualization
            if self.enable_visualization and self.save_plots:
                if self.iteration_counter % self.plot_interval == 0:
                    filename = self.visualizer.generate_plot(
                        trajectory,
                        self.profile_type,
                        self.generator.max_velocity,
                        self.generator.max_acceleration
                    )
                    if filename:
                        self.get_logger().info(f'Saved visualization to {filename}')
            
            self.iteration_counter += 1
            
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