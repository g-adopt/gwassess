"""Reference solution for Cockett et al. (2018) 3D Richards equation benchmark.

Reference:
    Cockett, R., Heagy, L. J., & Haber, E. (2018). Efficient 3D inversions using the
    Richards equation. Computers & Geosciences, 116, 91-102.
"""
from __future__ import division
from math import sin, tanh, exp


class CockettRichardsSolution3D(object):
    """Reference solution for Cockett et al. (2018) 3D heterogeneous benchmark.

    This is a numerical benchmark problem with spatially heterogeneous soil properties.
    The class provides the standard problem setup including initial conditions,
    boundary conditions, and the heterogeneous material property field.
    """

    def __init__(self, Lx=2.0, Ly=2.0, Lz=2.6):
        """Initialize Cockett 3D Richards reference solution.

        Args:
            Lx: Domain length in x-direction [L] (default: 2.0 m)
            Ly: Domain length in y-direction [L] (default: 2.0 m)
            Lz: Domain length in z-direction [L] (default: 2.6 m)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # Random seed points for heterogeneous field
        self.r = [0.0729, 0.0885, 0.7984, 0.9430, 0.6837,
                  0.1321, 0.7227, 0.1104, 0.1175, 0.6407]

    def heterogeneous_field(self, x, y, z):
        """Return heterogeneous indicator field I(x,y,z).

        This creates a spatially varying field that transitions smoothly between
        two material types using a combination of sinusoidal functions.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]
            z: z-coordinate [L]

        Returns:
            Indicator field I in [0, 1], where I=1 represents material 1
            and I=0 represents material 2
        """
        r = self.r
        indicator = (sin(3 * (x - r[0])) + sin(3 * (y - r[1]))
                     + sin(3 * (z - r[2])) + sin(3 * (x - r[3]))
                     + sin(3 * (y - r[4])) + sin(3 * (z - r[5]))
                     + sin(3 * (x - r[6])) + sin(3 * (y - r[7]))
                     + sin(3 * (z - r[8])))
        indicator = 0.5 * (1 + tanh(5 * indicator))
        return indicator

    def initial_condition(self, x, y, z):
        """Return initial pressure head.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]
            z: z-coordinate [L]

        Returns:
            Initial pressure head h [L]
        """
        return -3 + 2.9 * exp(5 * (z - self.Lz))

    def get_boundary_conditions(self):
        """Return boundary condition specification.

        Returns:
            Dictionary describing boundary conditions for each boundary ID.
            Format: {boundary_id: {'type': 'flux' or 'h', 'value': ...}}
        """
        # Standard boundary IDs for 3D box:
        # left=1, right=2, front=3, back=4, bottom=5, top=6
        return {
            'left': {'type': 'flux', 'value': 0.0},
            'right': {'type': 'flux', 'value': 0.0},
            'front': {'type': 'flux', 'value': 0.0},
            'back': {'type': 'flux', 'value': 0.0},
            'bottom': {'type': 'flux', 'value': 0.0},
            'top': {'type': 'h', 'value': -0.1}
        }

    def get_soil_parameters(self, x, y, z):
        """Return spatially varying soil hydraulic parameters.

        The parameters vary smoothly between two material types based on
        the heterogeneous indicator field.

        Args:
            x: x-coordinate [L]
            y: y-coordinate [L]
            z: z-coordinate [L]

        Returns:
            Dictionary with all soil parameters for Van Genuchten model.
        """
        indicator = self.heterogeneous_field(x, y, z)

        return {
            'model_type': 'VanGenuchten',
            'theta_r': 0.02 * indicator + 0.035 * (1 - indicator),
            'theta_s': 0.417 * indicator + 0.401 * (1 - indicator),
            'n': 1.592 * indicator + 1.474 * (1 - indicator),
            'alpha': 13.8 * indicator + 11.5 * (1 - indicator),
            'Ks': 5.82e-05 * indicator + 1.69e-05 * (1 - indicator),
            'Ss': 0.0
        }
