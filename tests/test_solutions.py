"""Unit tests for gwassess solution classes."""
import unittest
import numpy as np
from gwassess import TracyRichardsSolution2D, VauclinRichardsSolution2D, CockettRichardsSolution3D


class TestTracySolution(unittest.TestCase):
    """Test Tracy 2D analytical solution."""

    def setUp(self):
        """Set up Tracy solution with standard parameters."""
        self.solution = TracyRichardsSolution2D(
            alpha=0.328,
            hr=-15.24,
            L=15.24,
            theta_r=0.15,
            theta_s=0.45,
            Ks=1.0e-05
        )

    def test_instantiation(self):
        """Test that solution can be instantiated."""
        self.assertIsNotNone(self.solution)
        self.assertEqual(self.solution.alpha, 0.328)
        self.assertEqual(self.solution.L, 15.24)

    def test_pressure_head_specified_head(self):
        """Test pressure head evaluation with specified head BCs."""
        x, y, t = 7.62, 7.62, 1000.0
        h = self.solution.pressure_head_specified_head(x, y, t)
        self.assertIsInstance(h, (float, np.floating))
        # Pressure head should be negative (unsaturated)
        self.assertLess(h, 0)
        # Should be within reasonable range
        self.assertGreater(h, -20)

    def test_pressure_head_no_flux(self):
        """Test pressure head evaluation with no-flux BCs."""
        x, y, t = 7.62, 7.62, 1000.0
        h = self.solution.pressure_head_no_flux(x, y, t)
        self.assertIsInstance(h, (float, np.floating))
        self.assertLess(h, 0)
        self.assertGreater(h, -20)

    def test_pressure_head_cartesian(self):
        """Test Cartesian coordinate wrapper."""
        X = [7.62, 7.62]
        t = 1000.0
        h1 = self.solution.pressure_head_cartesian(X, t, bc_type='specified_head')
        h2 = self.solution.pressure_head_specified_head(X[0], X[1], t)
        self.assertAlmostEqual(h1, h2)

    def test_moisture_content(self):
        """Test moisture content calculation."""
        # Unsaturated
        theta_unsat = self.solution.moisture_content(-5.0)
        self.assertGreater(theta_unsat, self.solution.theta_r)
        self.assertLess(theta_unsat, self.solution.theta_s)

        # Saturated
        theta_sat = self.solution.moisture_content(0.5)
        self.assertEqual(theta_sat, self.solution.theta_s)


class TestVauclinSolution(unittest.TestCase):
    """Test Vauclin 2D reference solution."""

    def setUp(self):
        """Set up Vauclin solution with standard parameters."""
        self.solution = VauclinRichardsSolution2D()

    def test_instantiation(self):
        """Test that solution can be instantiated."""
        self.assertIsNotNone(self.solution)
        self.assertEqual(self.solution.Lx, 3.0)
        self.assertEqual(self.solution.Ly, 2.0)

    def test_initial_condition(self):
        """Test initial condition evaluation."""
        h0 = self.solution.initial_condition(1.5, 1.0)
        self.assertEqual(h0, -10.0)

    def test_top_boundary_flux(self):
        """Test top boundary flux evaluation."""
        q = self.solution.top_boundary_flux(0.0, 1000.0)
        self.assertIsInstance(q, (float, np.floating))
        # Flux should be positive (downward)
        self.assertGreaterEqual(q, 0)

    def test_get_boundary_conditions(self):
        """Test boundary condition retrieval."""
        bcs = self.solution.get_boundary_conditions(0.0)
        self.assertIsInstance(bcs, dict)
        self.assertIn('left', bcs)
        self.assertIn('top', bcs)

    def test_get_soil_parameters(self):
        """Test soil parameter retrieval."""
        params = self.solution.get_soil_parameters()
        self.assertIsInstance(params, dict)
        self.assertEqual(params['model_type'], 'Haverkamp')
        self.assertEqual(params['theta_r'], 0.10)


class TestCockettSolution(unittest.TestCase):
    """Test Cockett 3D reference solution."""

    def setUp(self):
        """Set up Cockett solution with standard parameters."""
        self.solution = CockettRichardsSolution3D()

    def test_instantiation(self):
        """Test that solution can be instantiated."""
        self.assertIsNotNone(self.solution)
        self.assertEqual(self.solution.Lx, 2.0)
        self.assertEqual(self.solution.Lz, 2.6)

    def test_heterogeneous_field(self):
        """Test heterogeneous field evaluation."""
        indicator = self.solution.heterogeneous_field(1.0, 1.0, 1.3)
        self.assertIsInstance(indicator, (float, np.floating))
        # Indicator field should be in [0, 1]
        self.assertGreaterEqual(indicator, 0)
        self.assertLessEqual(indicator, 1)

    def test_initial_condition(self):
        """Test initial condition evaluation."""
        h0 = self.solution.initial_condition(1.0, 1.0, 1.3)
        self.assertIsInstance(h0, (float, np.floating))
        # Should be negative (unsaturated)
        self.assertLess(h0, 0)

    def test_get_boundary_conditions(self):
        """Test boundary condition retrieval."""
        bcs = self.solution.get_boundary_conditions()
        self.assertIsInstance(bcs, dict)
        self.assertIn('left', bcs)
        self.assertIn('top', bcs)
        # Top boundary should have specified head
        self.assertEqual(bcs['top']['type'], 'h')

    def test_get_soil_parameters(self):
        """Test spatially varying soil parameters."""
        params = self.solution.get_soil_parameters(1.0, 1.0, 1.3)
        self.assertIsInstance(params, dict)
        self.assertEqual(params['model_type'], 'VanGenuchten')
        # Parameters should be positive
        self.assertGreater(params['theta_s'], params['theta_r'])
        self.assertGreater(params['Ks'], 0)


if __name__ == '__main__':
    unittest.main()
