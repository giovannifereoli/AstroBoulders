import numpy as np
import spiceypy as spice
from GravModels.Models.PointMass import PointMass
from GravModels.Filters.EKF_SNC import ExtendedKalmanFilterSNC
from GravModels.utils.MeasurementModels import RadioMetricModels
from GravModels.utils.Propagator import Propagator
from GravModels.CelestialBodies.Asteroids import Didymos
from GravModels.utils.Plotting import plot_asteroid_3d
from GravModels.utils.Plotting import plot_range_and_range_rate

## Initialization true trajectory and measurements

# Load SPICE kernels (replace with appropriate paths)
spice.furnsh("Kernels/de432s.bsp")  # SPICE planetary ephemeris
spice.furnsh("Kernels/didymos_hor_200101_300101_v01.bsp")  # SPICE kernel for Didymos
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
sigma_range = 1e-9
sigma_range_rate = 1e-5
measurement_model = RadioMetricModels(asteroid, t_eval, sigma_range, sigma_range_rate)

# Process the trajectory to get the real measurements
measurements_true = measurement_model.process_trajectory(trajectory)

for i, (range_measurement, range_rate_measurement) in enumerate(measurements_true):
    print(
        f"State {i}: Range = {range_measurement:.2f} km, Range Rate = {range_rate_measurement:.2f} km/sec"
    )

# Plot the range and range rate
plot_range_and_range_rate(t_eval, measurements_true)

# Plot the trajectory
plot_asteroid_3d(asteroid, disc=10, Trajectory=trajectory[:, :3])

## Initialization of the filter

# Define dynamics
point_mass = PointMass(asteroid)

# Initial conditions
initial_state_true = np.concatenate((initial_position, initial_velocity))
sigma_state = np.array([1e-6, 1e-6, 0, 1e-4, 1e-4, 0, 0.1, 0.1, 0])
initial_state_filter = np.concatenate(
    (initial_state_true, np.zeros(3))
) + np.random.normal(0, sigma_state)

# Time span
t_span = (0, 0.5)
num_points = 10000
t_eval = np.linspace(*t_span, num_points)

# Kalman filter setup
Q = np.diag([0.1**2, 0.1**2, 0.1**2])
R = np.diag([1e-9**2, 1e-5**2])
P0 = np.diag(np.square(sigma_state))
ekf = ExtendedKalmanFilterSNC(
    point_mass, measurement_model, Q, R, initial_state_filter, P0
)

# Run Kalman filter
filtered_state = np.zeros((len(initial_state_filter), len(t_eval) - 1))
covariance = np.zeros(
    (len(initial_state_filter), len(initial_state_filter), len(t_eval) - 1)
)
for i in range(len(t_eval) - 1):
    ekf.predict(t_eval[1] - t_eval[0])
    ekf.update(measurements_true[i])
    filtered_state[:, i] = ekf.x
    covariance[:, :, i] = ekf.P
    print(f"Filter Epoch [{i + 1}/{len(t_eval) - 1}]")

# Save filtered_state and covariance
np.save("filtered_state_EKF_CR3BP.npy", filtered_state)
np.save("covariance_EKF_CR3BP.npy", covariance)


## FOR DMC

# Initialize B matrix for DMC
# tau = 1  # Correlation time, in general orbital period is fine
# B = np.diag([1 / tau, 1 / tau, 1 / tau])  # Jacobian first-order Gauss-Markov process

# Calculate residuals
# resdyn_true = np.array(
#    [
#        bcr4bp_srp.cr3bp_residual(solution_true.t[i], solution_true.y[:, i])
#        for i in range(len(t_eval) - 1)
#    ]
# )
