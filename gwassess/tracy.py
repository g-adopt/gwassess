"""Analytical solutions for Tracy (2006) Richards equation benchmarks.

Provides 2D and 3D analytical solutions for the Richards equation with
exponential (Gardner) soil properties on square/cubic domains.

The exponential soil model uses:
    K(h) = Ks * exp(alpha * h)
    theta(h) = theta_r + (theta_s - theta_r) * exp(alpha * h)

Tracy tests three alpha values: 0.164, 0.328, 0.492 m^{-1}, representing
low, moderate, and high nonlinearity respectively.

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

        theta(h) = theta_r + (theta_s - theta_r) * exp(alpha * h)  for h <= 0
        theta(h) = theta_s                                         for h > 0

        Args:
            h: pressure head [L]

        Returns:
            Moisture content theta [-]
        """
        if h <= 0:
            theta = self.theta_r + (self.theta_s - self.theta_r) * exp(self.alpha * h)
        else:
            theta = self.theta_s
        return theta


class TracyRichardsSolution3D(object):
    """Analytical solution for 3D Richards equation in a cubic domain.

    Extends Tracy (2006) to three dimensions. The domain is a cube of side L
    with the vertical coordinate z pointing upward. Gravity acts in the -z
    direction. The solution uses the specified-head boundary condition case:
    h = h_r on all five faces except the top, where h is prescribed by the
    steady-state analytical expression.

    The 3D solution adds a second horizontal sine mode relative to the 2D
    case, giving sin(pi*x/L)*sin(pi*y/L) in the horizontal plane, and
    beta = sqrt(alpha^2/4 + 2*(pi/L)^2) to account for both horizontal
    wavenumbers.
    """

    def __init__(self, alpha, hr, L, theta_r, theta_s, Ks):
        """Initialize Tracy 3D Richards solution.

        Args:
            alpha: Exponential soil parameter [1/L]
            hr: Reference pressure head (typically -L) [L]
            L: Domain side length (cube L x L x L) [L]
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

        self.h0 = 1 - exp(alpha * hr)
        self.c = alpha * (theta_s - theta_r) / Ks

    def pressure_head(self, x, y, z, t):
        """Return pressure head for specified head boundary conditions.

        Specified head h = h_r on bottom and all four sides; the top face
        has h prescribed by the steady-state expression.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]
            z: z-coordinate (vertical, upward) [L]
            t: time [T]

        Returns:
            Pressure head h [L]
        """
        alpha = self.alpha
        L = self.L
        h0 = self.h0
        hr = self.hr
        c = self.c

        # 3D beta includes both horizontal wavenumbers
        beta = sqrt(alpha**2 / 4 + (pi / L)**2 + (pi / L)**2)

        # Steady-state solution
        hss = (h0 * sin(pi * x / L) * sin(pi * y / L)
               * exp((alpha / 2) * (L - z))
               * sinh(beta * z) / sinh(beta * L))

        # Transient correction (Fourier series in z)
        phi = 0
        for k in range(1, 200):
            lambdak = k * pi / L
            gamma = (beta**2 + lambdak**2) / c
            phi += ((-1)**k) * (lambdak / gamma) * sin(lambdak * z) * exp(-gamma * t)
        phi *= ((2 * h0) / (L * c)) * sin(pi * x / L) * sin(pi * y / L) * exp(alpha * (L - z) / 2)

        hBar = hss + phi

        # Transform back to pressure head
        hExact = (1 / alpha) * log(exp(alpha * hr) + hBar)

        return hExact

    def pressure_head_cartesian(self, X, t):
        """Return pressure head at a Cartesian location.

        Args:
            X: 3D Cartesian coordinates [x, y, z]
            t: time [T]

        Returns:
            Pressure head h [L]
        """
        return self.pressure_head(X[0], X[1], X[2], t)

    def steady_state_top_bc(self, x, y):
        """Return the steady-state pressure head at the top face (z = L).

        This is the Dirichlet boundary condition to apply at the top.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]

        Returns:
            Pressure head h at z = L [L]
        """
        alpha = self.alpha
        L = self.L
        h0 = self.h0
        hr = self.hr
        return (1 / alpha) * log(exp(alpha * hr) + h0 * sin(pi * x / L) * sin(pi * y / L))

    def moisture_content(self, h):
        """Return moisture content from pressure head using exponential model.

        theta(h) = theta_r + (theta_s - theta_r) * exp(alpha * h)  for h <= 0
        theta(h) = theta_s                                         for h > 0

        Args:
            h: pressure head [L]

        Returns:
            Moisture content theta [-]
        """
        if h <= 0:
            theta = self.theta_r + (self.theta_s - self.theta_r) * exp(self.alpha * h)
        else:
            theta = self.theta_s
        return theta
