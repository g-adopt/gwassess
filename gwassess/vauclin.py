"""Reference solution for Vauclin et al. (1979) 2D Richards equation benchmark.

Reference:
    Vauclin, M., Khanji, D., & Vachaud, G. (1979). Experimental and numerical study
    of a transient, two-dimensional unsaturated-saturated water table recharge problem.
    Water Resources Research, 15(5), 1089-1101.
    https://doi.org/10.1029/WR015i005p01089

The problem simulates recharge of a 2D water table in a domain of 3 x 2 metres.
The initial condition has the region z <= 0.65 m fully saturated (h = z - 0.65).
Boundary conditions:
  - Bottom and left: no flux (q · n = 0)
  - Right: fixed water table height (h = z - 0.65 m)
  - Top: water injection at 14.8 cm/hour for x <= 0.5 m, zero otherwise
The simulation runs for 8 hours.
"""
from __future__ import division
from math import tanh


class VauclinRichardsSolution2D(object):
    """Reference solution for Vauclin et al. (1979) 2D infiltration benchmark.

    This is a numerical benchmark problem without a closed-form analytical solution.
    The class provides the standard problem setup including initial conditions,
    boundary conditions, and soil parameters as specified in the original paper.

    The Haverkamp soil hydraulic model parameters are taken from:
        Haverkamp, R., et al. (1977). A comparison of numerical simulation models
        for one-dimensional infiltration. Soil Science Society of America Journal.
    """

    # Water table height from bottom of domain [m]
    WATER_TABLE_HEIGHT = 0.65

    # Infiltration rate from original paper: 14.8 cm/hr = 4.11e-05 m/s
    INFILTRATION_RATE = 14.8 / 100 / 3600  # m/s

    # Infiltration zone width: x <= 0.5 m
    INFILTRATION_WIDTH = 0.5

    # Simulation duration: 8 hours
    SIMULATION_DURATION = 8 * 3600  # seconds

    def __init__(self, Lx=3.0, Ly=2.0, theta_r=0.0, theta_s=0.30,
                 alpha=40000 / 100**2.90, beta=2.90,
                 A=2.99e6 / 100**5.0, gamma=5.0,
                 Ks=9.722e-05, Ss=0.0):
        """Initialize Vauclin 2D Richards reference solution.

        Default parameters match the original Vauclin (1979) paper, Table/Eq. 1
        (p. 1091) and Eq. 9 (p. 1093), converted from CGS (cm, hr) to SI (m, s).

        The Haverkamp retention and conductivity curves are:
            theta(h) = theta_s * alpha / (alpha + |h|^beta)
            K(h)     = Ks * A / (A + |h|^gamma)

        Paper values (h in cm): alpha = 40,000, beta = 2.90, A = 2.99e6,
        gamma = 5.0, theta_s = 0.30, Ks = 35 cm/hr. Since alpha and A carry
        units of [length]^exponent, the conversion to metres is:
            alpha_m = 40,000 / 100^2.90
            A_m     = 2.99e6 / 100^5.0

        Args:
            Lx: Domain length in x-direction [L] (default: 3.0 m)
            Ly: Domain length in y-direction [L] (default: 2.0 m)
            theta_r: Residual water content [-] (default: 0.0)
            theta_s: Saturated water content [-] (default: 0.30)
            alpha: Haverkamp retention parameter [m^beta] (default: 40000/100^2.90)
            beta: Haverkamp retention exponent [-] (default: 2.90)
            A: Haverkamp conductivity parameter [m^gamma] (default: 2.99e6/100^5.0)
            gamma: Haverkamp conductivity exponent [-] (default: 5.0)
            Ks: Saturated hydraulic conductivity [L/T] (default: 9.722e-05 m/s = 35 cm/hr)
            Ss: Specific storage coefficient [1/L] (default: 0.0)
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

        The initial condition represents a water table at z = 0.65 m from the bottom.
        Below the water table (y <= 0.65 m), the soil is fully saturated with h = y - 0.65.
        Above the water table, h < 0 (unsaturated). The 1.001 factor provides a slight
        offset to ensure the domain starts slightly unsaturated at the water table.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L] (vertical, 0 at bottom)

        Returns:
            Initial pressure head h [L]
        """
        return self.WATER_TABLE_HEIGHT - 1.001 * y

    def right_boundary_head(self, x, y):
        """Return pressure head at right boundary (fixed water table).

        The right boundary maintains the initial water table position.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]

        Returns:
            Pressure head h [L]
        """
        return self.WATER_TABLE_HEIGHT - 1.001 * y

    def top_boundary_flux(self, x, t):
        """Return top boundary flux as a function of position and time.

        This implements a time-dependent infiltration flux that is localized
        in space using smooth tanh transitions. Water is injected at a rate of
        14.8 cm/hour in the region x <= 0.5 m (using smooth transitions).

        Args:
            x: x-coordinate [L]
            t: time [T]

        Returns:
            Flux q [L/T] (positive into domain, i.e., downward)
        """
        # Time-dependent ramp-up (smooth start)
        time_factor = tanh(0.000125 * t)

        # Spatial localization: flux applied for x in [-0.5, 0.5]
        # Using smooth tanh transitions for numerical stability
        spatial_factor = (0.5 * (1 + tanh(10 * (x + self.INFILTRATION_WIDTH)))
                          - 0.5 * (1 + tanh(10 * (x - self.INFILTRATION_WIDTH))))

        return time_factor * self.INFILTRATION_RATE * spatial_factor

    def get_boundary_conditions(self, t=None):
        """Return boundary condition specification.

        From the original paper:
        - Bottom and left: no flux (q · n = 0)
        - Right: fixed water table height (h = z - 0.65 m)
        - Top: water injection at 14.8 cm/hour for x <= 0.5 m

        Args:
            t: current time [T] (optional, for compatibility)

        Returns:
            Dictionary describing boundary conditions for each boundary ID.
            Format: {boundary_id: {'type': 'flux' or 'h', 'value': ...}}
        """
        return {
            'left': {'type': 'flux', 'value': 0.0},
            'right': {'type': 'h', 'value': 'initial_condition'},
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
