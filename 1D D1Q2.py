import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Nx = 101              # Number of lattice points
Nt = 10000            # Number of time steps (enough for steady-state)
dx = 1.0              # Grid spacing
dt = 1.0              # Time step
alpha = 0.25          # Thermal diffusivity
tau = 0.5 + alpha/dt  # Relaxation time (LBM relation)

# Lattice velocities and weights for D1Q2
e = [-1, 1]
w = [0.5, 0.5]

# Initial condition: temperature array
T = np.zeros(Nx)
T[0] = 100.0  # Left boundary
T[-1] = 0.0   # Right boundary

# Distribution functions: f[direction, position]
f = np.zeros((2, Nx))
for i in range(2):
    f[i, :] = w[i] * T

# LBM Time Loop
for t in range(Nt):
    # Macroscopic variable: Temperature
    T = np.sum(f, axis=0)

    # Collision Step
    for i in range(2):
        feq = w[i] * T
        f[i, :] += -(f[i, :] - feq) / tau

    # Streaming Step
    f_temp = f.copy()
    f[0, 1:] = f_temp[0, :-1]  # Left-moving
    f[1, :-1] = f_temp[1, 1:]  # Right-moving

    # Boundary Conditions (Dirichlet)
    f[0, 0] = w[0] * 100.0
    f[1, 0] = w[1] * 100.0
    f[0, -1] = w[0] * 0.0
    f[1, -1] = w[1] * 0.0

# Final temperature
T = np.sum(f, axis=0)
x = np.linspace(0, 1, Nx)

# Analytical solution for comparison
T_analytical = 100 * (1 - x)

# --- Plotting ---
plt.plot(x, T, label='LBM-D1Q2', linewidth=2)
plt.plot(x, T_analytical, 'r--', label='Analytical', linewidth=2)
plt.xlabel("x")
plt.ylabel("Temperature")
plt.title("1D Heat Conduction: LBM vs Analytical")
plt.legend()
plt.grid(True)
plt.show()

# --- Error Analysis ---
abs_error = np.abs(T - T_analytical)
max_error = np.max(abs_error)
mean_error = np.mean(abs_error)

print(f"Max Absolute Error: {max_error:.6e}")
print(f"Mean Absolute Error: {mean_error:.6e}")
