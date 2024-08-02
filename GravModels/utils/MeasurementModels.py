import numpy as np
import spiceypy as spice

# TODO: generalize this class adding specific Ground Stations


class RadioMetricModels:
    def __init__(self, asteroid, epochs, noise_flag=True):
        """
        Initialize the RadioMetricModels with the asteroid and epochs.

        Parameters:
        - asteroid: An object representing the asteroid, with a 'body_name' attribute.
        - epochs: A list of epochs (in UTC) for which the position and velocity are computed.
        - noise_flag: Boolean flag to add noise to the measurements.
        """
        self.asteroid_name = asteroid.body_name
        self.epochs = epochs.tolist()
        self.noise_flag = noise_flag

        # Compute the positions and velocities of the asteroid with respect to Earth for all epochs
        self.pos_asteroid, self.vel_asteroid = self.compute_asteroid_states()

    def compute_asteroid_states(self):
        """
        Compute the positions and velocities of the asteroid with respect to Earth at the given epochs.

        Returns:
        - pos_asteroid: A list of 3-element arrays representing the position of the asteroid relative to Earth.
        - vel_asteroid: A list of 3-element arrays representing the velocity of the asteroid relative to Earth.
        """
        pos_asteroid = []
        vel_asteroid = []

        for epoch in self.epochs:
            state, _ = spice.spkezr(self.asteroid_name, epoch, "J2000", "NONE", "EARTH")
            pos_asteroid.append(state[:3])
            vel_asteroid.append(state[3:])

        return np.array(pos_asteroid), np.array(vel_asteroid)

    def compute_range(self, pos_sc, pos_asteroid):
        """
        Compute the range (distance) between the spacecraft and Earth.

        Parameters:
        - pos_sc: A 3-element array representing the position of the spacecraft relative to the asteroid.
        - pos_asteroid: A 3-element array representing the position of the asteroid relative to Earth.

        Returns:
        - range: The distance between the spacecraft and Earth.
        """
        pos_sc_earth = pos_asteroid + np.array(pos_sc)
        return np.linalg.norm(pos_sc_earth)

    def compute_range_rate(self, pos_sc, vel_sc, pos_asteroid, vel_asteroid):
        """
        Compute the range rate (rate of change of distance) between the spacecraft and Earth.

        Parameters:
        - pos_sc: A 3-element array representing the position of the spacecraft relative to the asteroid.
        - vel_sc: A 3-element array representing the velocity of the spacecraft relative to the asteroid.
        - pos_asteroid: A 3-element array representing the position of the asteroid relative to Earth.
        - vel_asteroid: A 3-element array representing the velocity of the asteroid relative to Earth.

        Returns:
        - range_rate: The rate of change of distance between the spacecraft and Earth.
        """
        pos_sc_earth = pos_asteroid + np.array(pos_sc)
        vel_sc_earth = vel_asteroid + np.array(vel_sc)

        range_vector = pos_sc_earth
        range_rate = np.dot(range_vector, vel_sc_earth) / np.linalg.norm(range_vector)
        return range_rate

    def add_noise(self, measurement, sigma):
        """
        Add Gaussian noise to a measurement.

        Parameters:
        - measurement: The measurement value.
        - sigma: The standard deviation of the noise.

        Returns:
        - noisy_measurement: The measurement with added Gaussian noise.
        """
        return measurement + np.random.normal(0, sigma)

    def get_measurements(
        self,
        pos_sc,
        vel_sc,
        epoch_idx,
        sigma_range=0,
        sigma_range_rate=0,
    ):
        """
        Get the range and range rate measurements for the spacecraft.

        Parameters:
        - pos_sc: A 3-element array representing the position of the spacecraft relative to the asteroid.
        - vel_sc: A 3-element array representing the velocity of the spacecraft relative to the asteroid.
        - pos_asteroid: A 3-element array representing the position of the asteroid relative to Earth.
        - vel_asteroid: A 3-element array representing the velocity of the asteroid relative to Earth.
        - sigma_range: Standard deviation of range noise.
        - sigma_range_rate: Standard deviation of range rate noise.

        Returns:
        - measurements: A tuple containing the range and range rate.
        """
        pos_asteroid = self.pos_asteroid[epoch_idx]
        vel_asteroid = self.vel_asteroid[epoch_idx]
        range_measurement = self.compute_range(pos_sc, pos_asteroid)
        range_rate_measurement = self.compute_range_rate(
            pos_sc, vel_sc, pos_asteroid, vel_asteroid
        )

        if self.noise_flag:
            range_measurement = self.add_noise(range_measurement, sigma_range)
            range_rate_measurement = self.add_noise(
                range_rate_measurement, sigma_range_rate
            )

        return range_measurement, range_rate_measurement

    def jacobian(self, pos_sc, vel_sc, epoch_idx):
        """
        Compute the Jacobian matrix of the range and range rate measurements.

        Parameters:
        - pos_sc: A 3-element array representing the position of the spacecraft relative to the asteroid.
        - vel_sc: A 3-element array representing the velocity of the spacecraft relative to the asteroid.
        - pos_asteroid: A 3-element array representing the position of the asteroid relative to Earth.
        - vel_asteroid: A 3-element array representing the velocity of the asteroid relative to Earth.

        Returns:
        - H: The Jacobian matrix.
        """
        pos_asteroid = self.pos_asteroid[epoch_idx]
        vel_asteroid = self.vel_asteroid[epoch_idx]
        pos_sc_earth = pos_asteroid + np.array(pos_sc)
        vel_sc_earth = vel_asteroid + np.array(vel_sc)
        range_ = np.linalg.norm(pos_sc_earth)
        range_rate = np.dot(pos_sc_earth, vel_sc_earth) / range_

        range_grad = np.concatenate((pos_sc_earth / range_, np.zeros(3)))
        range_rate_grad_position = (
            vel_sc_earth - pos_sc_earth * range_rate / range_
        ) / range_
        range_rate_grad_velocity = pos_sc_earth / range_
        range_rate_grad = np.concatenate(
            (range_rate_grad_position, range_rate_grad_velocity)
        )

        H = np.vstack((range_grad, range_rate_grad))
        return H

    def process_trajectory(self, trajectory, sigma_range=0, sigma_range_rate=0):
        """
        Process a trajectory of spacecraft states to compute range and range rate measurements.

        Parameters:
        - trajectory: A list of dictionaries, each containing 'position' and 'velocity' keys for the spacecraft.

        Returns:
        - measurements: A list of tuples, each containing the range and range rate for a state in the trajectory.
        """
        measurements = []
        for i in range(len(trajectory)):
            pos_sc = trajectory[i, :3]
            vel_sc = trajectory[i, 3:]
            epoch_idx = i
            range_measurement, range_rate_measurement = self.get_measurements(
                pos_sc,
                vel_sc,
                epoch_idx,
                sigma_range,
                sigma_range_rate,
            )
            measurements.append((range_measurement, range_rate_measurement))
        return measurements
