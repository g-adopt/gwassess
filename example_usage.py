#!/usr/bin/env python3
"""Example usage of gwassess package."""
import gwassess

print("=" * 70)
print("gwassess - Analytical Solutions for Richards Equation")
print("=" * 70)

# Example 1: Tracy 2D analytical solution
print("\n1. Tracy 2D Analytical Solution")
print("-" * 70)

tracy = gwassess.TracyRichardsSolution2D(
    alpha=0.328,      # Exponential soil parameter [1/m]
    hr=-15.24,        # Reference pressure head [m]
    L=15.24,          # Domain size [m]
    theta_r=0.15,     # Residual water content [-]
    theta_s=0.45,     # Saturated water content [-]
    Ks=1.0e-05        # Saturated hydraulic conductivity [m/s]
)

# Evaluate at a point
x, y, t = 7.62, 7.62, 1000.0
h_specified = tracy.pressure_head_specified_head(x, y, t)
h_noflux = tracy.pressure_head_no_flux(x, y, t)

print(f"At location ({x}, {y}) m, time {t} s:")
print(f"  Pressure head (specified BC): {h_specified:.6f} m")
print(f"  Pressure head (no-flux BC):   {h_noflux:.6f} m")

# Example 2: Vauclin 2D reference solution
print("\n2. Vauclin 2D Reference Solution")
print("-" * 70)

vauclin = gwassess.VauclinRichardsSolution2D()
h0 = vauclin.initial_condition(1.5, 1.0)
q_top = vauclin.top_boundary_flux(0.0, 5000.0)
params = vauclin.get_soil_parameters()

print(f"Initial condition: h0 = {h0} m")
print(f"Top boundary flux at t=5000s: q = {q_top:.6e} m/s")
print(f"Soil model: {params['model_type']}")
print(f"  theta_r = {params['theta_r']}, theta_s = {params['theta_s']}")

# Example 3: Cockett 3D reference solution
print("\n3. Cockett 3D Reference Solution")
print("-" * 70)

cockett = gwassess.CockettRichardsSolution3D()
x, y, z = 1.0, 1.0, 1.3
indicator = cockett.heterogeneous_field(x, y, z)
h0 = cockett.initial_condition(x, y, z)
params = cockett.get_soil_parameters(x, y, z)

print(f"At location ({x}, {y}, {z}) m:")
print(f"  Heterogeneous indicator: I = {indicator:.4f}")
print(f"  Initial condition: h0 = {h0:.6f} m")
print(f"  Soil model: {params['model_type']}")
print(f"  theta_r = {params['theta_r']:.4f}, theta_s = {params['theta_s']:.4f}")
print(f"  Ks = {params['Ks']:.6e} m/s")

print("\n" + "=" * 70)
print("All examples completed successfully!")
print("=" * 70)
