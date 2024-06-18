import numpy as np
import matplotlib.pyplot as plt
from GravModels.utils.Propagator import *
from GravModels.utils.Plotting import plot_asteroid_3d

# Example usage
if __name__ == "__main__":
    from GravModels.CelestialBodies.Asteroids import Didymos

    # Define initial conditions for the spacecraft
    initial_position = np.array([700, 0, 0])
    initial_velocity = np.array([0, 0.25, 0])  # initial velocity in km/s

    # Define time span for the integration
    t_span = (0, 60 * 60 * 24)  # simulate for one week
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Create an instance of the Propagator class with different models
    asteroid = Didymos()

    model = "Pines"
    print(f"Using model: {model}")
    propagator = Propagator(
        asteroid,
        initial_position,
        initial_velocity,
        t_span,
        t_eval,
        model_type=model,
    )

    # Propagate the equations of motion
    solution = propagator.propagate()

    # Extract the results
    trajectory = solution.y.T

    # Plot the trajectory
    plot_asteroid_3d(asteroid, disc=10, Trajectory=trajectory)
