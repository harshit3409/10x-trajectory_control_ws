#!/usr/bin/env python3
"""
Unit tests for Trajectory Generator
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajectory_controller.trajectory_generator import TrajectoryGenerator


class TestTrajectoryGenerator(unittest.TestCase):
    """Test cases for TrajectoryGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TrajectoryGenerator(max_velocity=0.5, max_acceleration=0.3)
    
    def test_generate_trajectory_trapezoidal(self):
        """Test trajectory generation with trapezoidal profile."""
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        trajectory = self.generator.generate_trajectory(path, 'trapezoidal')
        
        self.assertEqual(len(trajectory), len(path))
        self.assertEqual(len(trajectory[0]), 3)  # (x, y, t)
        
        # Check time is increasing
        times = [t for _, _, t in trajectory]
        self.assertTrue(all(times[i] <= times[i+1] for i in range(len(times)-1)))
    
    def test_generate_trajectory_constant(self):
        """Test trajectory generation with constant profile."""
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        trajectory = self.generator.generate_trajectory(path, 'constant')
        
        self.assertEqual(len(trajectory), len(path))
    
    def test_invalid_profile_type(self):
        """Test error handling with invalid profile type."""
        path = [(0.0, 0.0), (1.0, 0.0)]
        
        with self.assertRaises(ValueError):
            self.generator.generate_trajectory(path, 'invalid')
    
    def test_insufficient_points(self):
        """Test error handling with insufficient points."""
        path = [(0.0, 0.0)]
        
        with self.assertRaises(ValueError):
            self.generator.generate_trajectory(path)
    
    def test_get_velocity_at_time(self):
        """Test velocity retrieval at specific time."""
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        trajectory = self.generator.generate_trajectory(path, 'constant')
        
        mid_time = trajectory[1][2]
        velocity, _ = self.generator.get_velocity_at_time(trajectory, mid_time)
        
        self.assertGreaterEqual(velocity, 0.0)


if __name__ == '__main__':
    unittest.main()