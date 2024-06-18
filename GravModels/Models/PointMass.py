import numpy as np


class PointMass:
    def __init__(self, asteroid):
        """
        Initializes a PointMass object.

        Parameters:
        asteroid (Asteroid): An instance of the Asteroid class.

        Attributes:
        mu (float): The gravitational parameter of the asteroid.
        """
        self.mu = asteroid.mu

    def calculate_potential(self, point):
        """
        Calculates the gravitational potential due to a point mass at a given point.

        Parameters:
        - point: A numpy array representing the coordinates of the point in 3D space.

        Returns:
        - potential: The gravitational potential at the given point.
        """
        radius = np.sqrt(point[0][0] ** 2 + point[0][1] ** 2 + point[0][2] ** 2)
        potential = -self.mu / radius
        return potential

    def calculate_acceleration(self, point):
        """
        Calculates the acceleration due to a point mass at a given point.

        Parameters:
        - point: A numpy array representing the coordinates of the point.

        Returns:
        - acceleration: A numpy array representing the acceleration vector.
        """
        radius = np.sqrt(point[0][0] ** 2 + point[0][1] ** 2 + point[0][2] ** 2)
        acceleration = (
            -self.mu / radius**3 * np.array([point[0][0], point[0][1], point[0][2]])
        )
        return acceleration
