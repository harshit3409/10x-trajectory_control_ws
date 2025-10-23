#!/usr/bin/env python3
"""
Unit tests for Path Smoother
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trajectory_controller.path_smoother import PathSmoother


class TestPathSmoother(unittest.TestCase):
    """Test cases for PathSmoother class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.smoother = PathSmoother(smoothing_factor=0.0, num_points=50)
    
    def test_smooth_path_basic(self):
        """Test basic path smoothing."""
        waypoints = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 1.0)]
        smoothed = self.smoother.smooth_path(waypoints)
        
        self.assertEqual(len(smoothed), 50)
        self.assertIsInstance(smoothed[0], tuple)
        self.assertEqual(len(smoothed[0]), 2)
    
    def test_smooth_path_minimum_points(self):
        """Test with minimum number of waypoints."""
        waypoints = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]
        smoothed = self.smoother.smooth_path(waypoints)
        
        self.assertEqual(len(smoothed), 50)
    
    def test_smooth_path_insufficient_points(self):
        """Test error handling with insufficient points."""
        waypoints = [(0.0, 0.0), (1.0, 0.0)]
        
        with self.assertRaises(ValueError):
            self.smoother.smooth_path(waypoints)
    
    def test_path_length_calculation(self):
        """Test path length calculation."""
        path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        length = self.smoother.calculate_path_length(path)
        
        self.assertAlmostEqual(length, 2.0, places=5)
    
    def test_curvature_calculation(self):
        """Test curvature calculation."""
        # Create a circular path
        import numpy as np
        angles = np.linspace(0, np.pi/2, 20)
        radius = 1.0
        path = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
        
        curvatures = self.smoother.calculate_curvature(path)
        
        self.assertEqual(len(curvatures), len(path))
        # Curvature of circle should be approximately 1/radius
        avg_curvature = sum(curvatures) / len(curvatures)
        self.assertGreater(avg_curvature, 0.5)


if __name__ == '__main__':
    unittest.main()