import numpy as np
import spiceypy as spice
from GravModels.utils.Propagator import *
from GravModels.utils.MeasurementModels import *
from GravModels.utils.Plotting import plot_asteroid_3d
from GravModels.utils.Plotting import plot_range_and_range_rate

# TODO: better unit of measurement handling, asteroids seem smaller

# Example usage
if __name__ == "__main__":
    from GravModels.CelestialBodies.Asteroids import Didymos

    # Load SPICE kernels (replace with appropriate paths)
    spice.furnsh("Kernels/de432s.bsp")  # SPICE planetary ephemeris
    spice.furnsh(
        "Kernels/didymos_hor_200101_300101_v01.bsp"
    )  # SPICE kernel for Didymos
    spice.furnsh("Kernels/naif0012.tls")  # Leap seconds kernel

    # Define initial conditions for the spacecraft
    initial_position = np.array([700, 0, 0])
    initial_velocity = np.array([0, 0.25, 0])  # initial velocity in km/s

    # Define time span for the integration
    et = spice.str2et("2028-Jan-04 12:00:00")
    t_span = (et, et + 60 * 60 * 24 * 7)  # simulate for one week
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

    # Create an instance of the MeasurementModel class
    measurement_model = MeasurementModel(asteroid, t_eval)

    # Process the trajectory to get the measurements
    measurements = measurement_model.process_trajectory(trajectory)

    for i, (range_measurement, range_rate_measurement) in enumerate(measurements):
        print(
            f"State {i}: Range = {range_measurement:.2f} km, Range Rate = {range_rate_measurement:.2f} km/sec"
        )

    # Plot the range and range rate
    plot_range_and_range_rate(t_eval, measurements)

    # Plot the trajectory
    plot_asteroid_3d(asteroid, disc=10, Trajectory=trajectory[:, :3])
