import numpy as np
import spiceypy as spice


class MeasurementModel:
    def __init__(self, asteroid, epochs):
        """
        Initialize the MeasurementModel with the asteroid and epochs.

        Parameters:
        - asteroid: An object representing the asteroid, with a 'body_name' attribute.
        - epochs: A list of epochs (in UTC) for which the position and velocity are computed.
        """
        self.asteroid_name = asteroid.body_name
        self.epochs = epochs.tolist()

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
        range_rate = np.abs(
            np.dot(range_vector, vel_sc_earth) / np.linalg.norm(range_vector)
        )
        return range_rate

    def get_measurements(self, pos_sc, vel_sc, pos_asteroid, vel_asteroid):
        """
        Get the range and range rate measurements for the spacecraft.

        Parameters:
        - pos_sc: A 3-element array representing the position of the spacecraft relative to the asteroid.
        - vel_sc: A 3-element array representing the velocity of the spacecraft relative to the asteroid.
        - pos_asteroid: A 3-element array representing the position of the asteroid relative to Earth.
        - vel_asteroid: A 3-element array representing the velocity of the asteroid relative to Earth.

        Returns:
        - measurements: A tuple containing the range and range rate.
        """
        range_measurement = self.compute_range(pos_sc, pos_asteroid)
        range_rate_measurement = self.compute_range_rate(
            pos_sc, vel_sc, pos_asteroid, vel_asteroid
        )
        return range_measurement, range_rate_measurement

    def process_trajectory(self, trajectory):
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
            pos_asteroid = self.pos_asteroid[i]
            vel_asteroid = self.vel_asteroid[i]
            range_measurement, range_rate_measurement = self.get_measurements(
                pos_sc, vel_sc, pos_asteroid, vel_asteroid
            )
            measurements.append((range_measurement, range_rate_measurement))
        return measurements
