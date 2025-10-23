"""Reference solution for Vauclin et al. (1979) 2D Richards equation benchmark.

Reference:
    Vauclin, M., Khanji, D., & Vachaud, G. (1979). Experimental and numerical study
    of a transient, two-dimensional unsaturated-saturated water table recharge problem.
    Water Resources Research, 15(5), 1089-1101.
"""
from __future__ import division
from math import tanh


class VauclinRichardsSolution2D(object):
    """Reference solution for Vauclin et al. (1979) 2D infiltration benchmark.

    This is a numerical benchmark problem without a closed-form analytical solution.
    The class provides the standard problem setup including initial conditions,
    boundary conditions, and soil parameters as specified in the original paper.
    """

    def __init__(self, Lx=3.0, Ly=2.0, theta_r=0.10, theta_s=0.37,
                 alpha=0.44, beta=1.2924, A=0.0104, gamma=1.5722,
                 Ks=5e-05, Ss=1e-05):
        """Initialize Vauclin 2D Richards reference solution.

        Args:
            Lx: Domain length in x-direction [L] (default: 3.0 m)
            Ly: Domain length in y-direction [L] (default: 2.0 m)
            theta_r: Residual water content [-] (default: 0.10)
            theta_s: Saturated water content [-] (default: 0.37)
            alpha: Haverkamp model parameter (default: 0.44)
            beta: Haverkamp model parameter (default: 1.2924)
            A: Haverkamp model parameter (default: 0.0104)
            gamma: Haverkamp model parameter (default: 1.5722)
            Ks: Saturated hydraulic conductivity [L/T] (default: 5e-05 m/s)
            Ss: Specific storage coefficient [1/L] (default: 1e-05 1/m)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.gamma = gamma
        self.Ks = Ks
        self.Ss = Ss

    def initial_condition(self, x, y):
        """Return initial pressure head.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]

        Returns:
            Initial pressure head h [L]
        """
        return -10.0

    def top_boundary_flux(self, x, t):
        """Return top boundary flux as a function of position and time.

        This implements a time-dependent infiltration flux that is localized
        in space using smooth tanh transitions.

        Args:
            x: x-coordinate [L]
            t: time [T]

        Returns:
            Flux q [L/T] (positive downward)
        """
        # Time-dependent ramp-up
        time_factor = tanh(0.0005 * t)

        # Spatial localization: flux applied in central region
        spatial_factor = (0.5 * (1 + tanh(10 * (x + 0.5)))
                          - 0.5 * (1 + tanh(10 * (x - 0.5))))

        return time_factor * 2e-05 * spatial_factor

    def get_boundary_conditions(self, t):
        """Return boundary condition specification.

        Args:
            t: current time [T]

        Returns:
            Dictionary describing boundary conditions for each boundary ID.
            Format: {boundary_id: {'type': 'flux' or 'h', 'value': ...}}
        """
        # Standard boundary IDs: left=1, right=2, bottom=3, top=4
        return {
            'left': {'type': 'flux', 'value': 0.0},
            'right': {'type': 'flux', 'value': 0.0},
            'bottom': {'type': 'flux', 'value': 0.0},
            'top': {'type': 'flux', 'value': 'time_and_space_dependent'}
        }

    def get_soil_parameters(self):
        """Return soil hydraulic parameters as a dictionary.

        Returns:
            Dictionary with all soil parameters for Haverkamp model.
        """
        return {
            'model_type': 'Haverkamp',
            'theta_r': self.theta_r,
            'theta_s': self.theta_s,
            'alpha': self.alpha,
            'beta': self.beta,
            'A': self.A,
            'gamma': self.gamma,
            'Ks': self.Ks,
            'Ss': self.Ss
        }
