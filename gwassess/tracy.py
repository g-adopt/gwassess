"""Analytical solutions for Tracy (2006) 2D Richards equation benchmark.

Reference:
    Tracy, F. T. (2006). Clean two- and three-dimensional analytical solutions of
    Richards' equation for testing numerical solvers. Water Resources Research, 42(8).
    https://doi.org/10.1029/2005WR004638
"""
from __future__ import division
from math import sqrt, sin, cos, exp, sinh, pi, log


class TracyRichardsSolution2D(object):
    """Analytical solution for 2D Richards equation in a square domain.

    This implements the analytical solutions from Tracy (2006) for steady-state
    and transient flow in unsaturated porous media with exponential soil properties.

    Two boundary condition cases are provided:
    1. Specified head on all boundaries
    2. No-flux on lateral boundaries with specified head on top/bottom
    """

    def __init__(self, alpha, hr, L, theta_r, theta_s, Ks):
        """Initialize Tracy 2D Richards solution.

        Args:
            alpha: Exponential soil parameter [1/L]
            hr: Reference pressure head (typically negative) [L]
            L: Domain size (square domain L x L) [L]
            theta_r: Residual water content [-]
            theta_s: Saturated water content [-]
            Ks: Saturated hydraulic conductivity [L/T]
        """
        self.alpha = alpha
        self.hr = hr
        self.L = L
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.Ks = Ks

        # Precompute commonly used values
        self.h0 = 1 - exp(alpha * hr)
        self.c = alpha * (theta_s - theta_r) / Ks

    def pressure_head_specified_head(self, x, y, t):
        """Return pressure head for specified head boundary conditions.

        This is the analytical solution from Tracy (2006), page 4, equation (15-17).
        Specified head boundary conditions are applied on all four boundaries.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]
            t: time [T]

        Returns:
            Pressure head h [L]
        """
        alpha = self.alpha
        L = self.L
        h0 = self.h0
        hr = self.hr
        c = self.c

        # Steady-state solution
        beta = sqrt(alpha**2 / 4 + (pi / L)**2)
        hss = h0 * sin(pi * x / L) * exp((alpha / 2) * (L - y)) * sinh(beta * y) / sinh(beta * L)

        # Transient correction term
        phi = 0
        for k in range(1, 200):
            lambdak = k * pi / L
            gamma = (beta**2 + lambdak**2) / c
            phi += ((-1)**k) * (lambdak / gamma) * sin(lambdak * y) * exp(-gamma * t)
        phi *= ((2 * h0) / (L * c)) * sin(pi * x / L) * exp(alpha * (L - y) / 2)

        hBar = hss + phi

        # Transform back to pressure head
        hExact = (1 / alpha) * log(exp(alpha * hr) + hBar)

        return hExact

    def pressure_head_no_flux(self, x, y, t):
        """Return pressure head for no-flux lateral boundary conditions.

        This is the analytical solution from Tracy (2006), page 5, equation (18-20).
        No-flux boundary conditions on left/right, specified head on top/bottom.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]
            t: time [T]

        Returns:
            Pressure head h [L]
        """
        alpha = self.alpha
        L = self.L
        h0 = self.h0
        hr = self.hr
        c = self.c

        # Steady-state solution
        beta = sqrt(alpha**2 / 4 + (2 * pi / L)**2)
        hss = (h0 / 2) * exp((alpha / 2) * (L - y)) * (
            sinh(alpha * y / 2) / sinh(alpha * L / 2)
            - cos(2 * pi * x / L) * sinh(beta * y) / sinh(beta * L)
        )

        # Transient correction term
        phi = 0
        for k in range(1, 200):
            lambdak = k * pi / L
            gamma1 = (lambdak**2 + alpha**2 / 4) / c
            gamma2 = ((2 * pi / L)**2 + lambdak**2 + alpha**2 / 4) / c
            phi += ((-1)**k) * lambdak * (
                (1 / gamma1) * exp(-gamma1 * t)
                - (1 / gamma2) * cos(2 * pi * x / L) * exp(-gamma2 * t)
            ) * sin(lambdak * y)
        phi *= (h0 / (L * c)) * exp(alpha * (L - y) / 2)

        hBar = hss + phi

        # Transform back to pressure head
        hExact = (1 / alpha) * log(exp(alpha * hr) + hBar)

        return hExact

    def pressure_head_cartesian(self, X, t, bc_type='specified_head'):
        """Return pressure head at Cartesian location.

        Args:
            X: 2D Cartesian coordinates [x, y]
            t: time [T]
            bc_type: 'specified_head' or 'no_flux'

        Returns:
            Pressure head h [L]
        """
        x, y = X[0], X[1]
        if bc_type == 'specified_head':
            return self.pressure_head_specified_head(x, y, t)
        elif bc_type == 'no_flux':
            return self.pressure_head_no_flux(x, y, t)
        else:
            raise ValueError("bc_type must be 'specified_head' or 'no_flux'")

    def moisture_content(self, h):
        """Return moisture content from pressure head using exponential model.

        Args:
            h: pressure head [L]

        Returns:
            Moisture content theta [-]
        """
        if h <= 0:
            theta = self.theta_r + self.alpha * (self.theta_s - self.theta_r) / (
                self.alpha + abs(h)**1.0
            )
        else:
            theta = self.theta_s
        return theta
