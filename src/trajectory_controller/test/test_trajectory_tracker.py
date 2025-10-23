#!/usr/bin/env python3
"""
Unit tests for Trajectory Tracker
"""

import unittest
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajectory_controller.trajectory_tracker import PIDController, TrajectoryTracker


class TestPIDController(unittest.TestCase):
    """Test cases for PIDController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pid = PIDController(kp=1.0, ki=0.1, kd=0.05, output_limit=1.0)
    
    def test_proportional_response(self):
        """Test proportional term response."""
        error = 0.5
        output = self.pid.compute(error, 0.0)
        
        self.assertGreater(output, 0.0)
        self.assertLessEqual(abs(output), 1.0)
    
    def test_output_limits(self):
        """Test output limiting."""
        large_error = 100.0
        output = self.pid.compute(large_error, 0.0)
        
        self.assertLessEqual(abs(output), 1.0)
    
    def test_reset(self):
        """Test controller reset."""
        self.pid.compute(1.0, 0.0)
        self.pid.reset()
        
        self.assertEqual(self.pid.integral, 0.0)
        self.assertEqual(self.pid.previous_error, 0.0)


class TestTrajectoryTracker(unittest.TestCase):
    """Test cases for TrajectoryTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = TrajectoryTracker(
            linear_gains=(1.0, 0.0, 0.1),
            angular_gains=(2.0, 0.0, 0.2),
            max_linear_velocity=0.5,
            max_angular_velocity=1.0
        )
    
    def test_set_trajectory(self):
        """Test setting a new trajectory."""
        trajectory = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 2.0)]
        self.tracker.set_trajectory(trajectory)
        
        self.assertEqual(len(self.tracker.trajectory), 3)
    
    def test_get_target_pose(self):
        """Test getting target pose at specific time."""
        trajectory = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 2.0)]
        self.tracker.set_trajectory(trajectory)
        
        x, y, yaw = self.tracker.get_target_pose(0.5)
        
        self.assertAlmostEqual(x, 0.5, places=1)
        self.assertAlmostEqual(y, 0.0, places=1)
    
    def test_compute_control(self):
        """Test control computation."""
        trajectory = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 2.0)]
        self.tracker.set_trajectory(trajectory)
        
        current_pose = (0.0, 0.0, 0.0)
        linear_vel, angular_vel = self.tracker.compute_control(current_pose, 0.0)
        
        self.assertIsInstance(linear_vel, float)
        self.assertIsInstance(angular_vel, float)
    
    def test_normalize_angle(self):
        """Test angle normalization."""
        angle1 = self.tracker._normalize_angle(3.5 * math.pi)
        angle2 = self.tracker._normalize_angle(-3.5 * math.pi)
        
        self.assertLessEqual(abs(angle1), math.pi)
        self.assertLessEqual(abs(angle2), math.pi)


if __name__ == '__main__':
    unittest.main()