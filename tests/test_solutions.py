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
        """Test initial condition evaluation.

        The initial condition represents a water table at z = 0.65 m.
        h(x, y) = 0.65 - 1.001 * y
        """
        # At y = 0 (bottom), h should be slightly less than 0.65
        h_bottom = self.solution.initial_condition(1.5, 0.0)
        self.assertAlmostEqual(h_bottom, 0.65, places=2)

        # At y = 1.0 m, h should be approximately -0.351
        h_mid = self.solution.initial_condition(1.5, 1.0)
        expected = 0.65 - 1.001 * 1.0
        self.assertAlmostEqual(h_mid, expected, places=5)

        # At y = 2.0 m (top), h should be approximately -1.352
        h_top = self.solution.initial_condition(1.5, 2.0)
        expected_top = 0.65 - 1.001 * 2.0
        self.assertAlmostEqual(h_top, expected_top, places=5)

    def test_right_boundary_head(self):
        """Test right boundary head (fixed water table)."""
        h_bc = self.solution.right_boundary_head(3.0, 1.0)
        h_ic = self.solution.initial_condition(3.0, 1.0)
        self.assertEqual(h_bc, h_ic)

    def test_top_boundary_flux(self):
        """Test top boundary flux evaluation."""
        # At x=0 (center of infiltration zone), flux should be non-zero
        q_center = self.solution.top_boundary_flux(0.0, 10000.0)
        self.assertIsInstance(q_center, (float, np.floating))
        # Flux should be positive (into domain)
        self.assertGreater(q_center, 0)

        # At x=2.0 (outside infiltration zone), flux should be near zero
        q_far = self.solution.top_boundary_flux(2.0, 10000.0)
        self.assertLess(q_far, q_center * 0.01)  # Much smaller than center

    def test_get_boundary_conditions(self):
        """Test boundary condition retrieval."""
        bcs = self.solution.get_boundary_conditions(0.0)
        self.assertIsInstance(bcs, dict)
        self.assertIn('left', bcs)
        self.assertIn('top', bcs)
        # Right boundary should have specified head (water table)
        self.assertEqual(bcs['right']['type'], 'h')
        # Left and bottom should be no-flux
        self.assertEqual(bcs['left']['type'], 'flux')
        self.assertEqual(bcs['bottom']['type'], 'flux')

    def test_get_soil_parameters(self):
        """Test soil parameter retrieval."""
        params = self.solution.get_soil_parameters()
        self.assertIsInstance(params, dict)
        self.assertEqual(params['model_type'], 'Haverkamp')
        # Updated parameters matching Vauclin 1979 paper
        self.assertEqual(params['theta_r'], 0.01)
        self.assertEqual(params['theta_s'], 0.37)
        self.assertAlmostEqual(params['Ks'], 9.722e-05, places=8)

    def test_class_constants(self):
        """Test class-level constants from paper."""
        self.assertEqual(VauclinRichardsSolution2D.WATER_TABLE_HEIGHT, 0.65)
        self.assertAlmostEqual(
            VauclinRichardsSolution2D.INFILTRATION_RATE,
            14.8 / 100 / 3600,  # 14.8 cm/hr in m/s
            places=8
        )
        self.assertEqual(VauclinRichardsSolution2D.INFILTRATION_WIDTH, 0.5)
        self.assertEqual(VauclinRichardsSolution2D.SIMULATION_DURATION, 8 * 3600)


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
