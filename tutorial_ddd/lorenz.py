import numpy as np

def lorenz_system(xyz, sigma, rho, beta):
    """
    Defines the Lorenz system of differential equations.

    Args:
        xyz (list or np.array): A list or array containing the current x, y, and z values.
        sigma (float): The sigma parameter of the Lorenz system.
        rho (float): The rho parameter of the Lorenz system.
        beta (float): The beta parameter of the Lorenz system.

    Returns:
        np.array: An array containing the calculated dx/dt, dy/dt, and dz/dt values.
    """
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def simulate_lorenz(
    initial_conditions,
    sigma=10.0,
    rho=28.0,
    beta=8/3,
    dt=0.01,
    num_steps=10000
):
    """
    Simulates the Lorenz system using the Runge-Kutta 4 (RK4) method.

    Args:
        initial_conditions (list or np.array): Initial [x, y, z] values.
        sigma (float): The sigma parameter.
        rho (float): The rho parameter.
        beta (float): The beta parameter.
        dt (float): The time step for the simulation.
        num_steps (int): The total number of simulation steps.

    Returns:
        np.array: A 2D array where each row is [x, y, z] at a given time step.
    """
    # Initialize an array to store the results
    trajectory = np.zeros((num_steps, 3))
    trajectory[0] = initial_conditions
   
    derivatives = np.zeros((num_steps, 3))

    # Current state
    current_xyz = np.array(initial_conditions)

    for i in range(1, num_steps):
        # Runge-Kutta 4 (RK4) method
        k1 = dt * lorenz_system(current_xyz, sigma, rho, beta)
        k2 = dt * lorenz_system(current_xyz + 0.5 * k1, sigma, rho, beta)
        k3 = dt * lorenz_system(current_xyz + 0.5 * k2, sigma, rho, beta)
        k4 = dt * lorenz_system(current_xyz + k3, sigma, rho, beta)

        current_xyz = current_xyz + (k1 + 2*k2 + 2*k3 + k4) / 6
        trajectory[i] = current_xyz
        derivatives[i-1] = (k1 + 2*k2 + 2*k3 + k4) / 6

    # calculate derivative of last time step
    # Runge-Kutta 4 (RK4) method
    k1 = dt * lorenz_system(current_xyz, sigma, rho, beta)
    k2 = dt * lorenz_system(current_xyz + 0.5 * k1, sigma, rho, beta)
    k3 = dt * lorenz_system(current_xyz + 0.5 * k2, sigma, rho, beta)
    k4 = dt * lorenz_system(current_xyz + k3, sigma, rho, beta)
    derivatives[i] = (k1 + 2*k2 + 2*k3 + k4) / 6

    return trajectory, derivatives

