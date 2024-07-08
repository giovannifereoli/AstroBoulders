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

    def dynamics(self, t, y):
        """
        Computes the dynamics of the system at a given time and state.

        Parameters:
        - t (float): The current time.
        - y (list): The current state vector [x, y, z, vx, vy, vz].

        Returns:
        - list: The derivative of the state vector [vx, vy, vz, ax, ay, az].
        """
        x, y, z, vx, vy, vz = y
        r = np.array([[x, y, z]])
        a = self.calculate_acceleration(r)
        ax, ay, az = a[0]
        return [vx, vy, vz, ax, ay, az]

    def dynamics_stm(self, t, y):
        """
        Computes the dynamics of the state transition matrix (STM) along with the state vector.

        Parameters:
        - t (float): The current time.
        - y (list): The state vector containing the position and velocity components along with the STM.

        Returns:
        - list: The derivative of the state vector along with the derivative of the STM, flattened into a list.
        """
        stm = np.reshape(y[6:], (6, 6))
        x, y, z, vx, vy, vz = y[:6]

        r = np.array([[x, y, z]])
        a = self.calculate_acceleration(r)
        ax, ay, az = a[0]

        A = self.jacobian(t, np.array([x, y, z, vx, vy, vz]))
        d_stm = np.dot(A, stm)

        return [vx, vy, vz, ax, ay, az] + d_stm.flatten().tolist()

    def jacobian(self, t, y):
        """
        Compute the Jacobian matrix for the point mass system.

        Parameters:
        - t: Current time
        - y: State vector containing the position and velocity components (x, y, z, vx, vy, vz)

        Returns:
        - A: Jacobian matrix
        """
        x, y, z, vx, vy, vz = y
        r = np.sqrt(x**2 + y**2 + z**2)
        mu_r3 = self.mu / r**3
        mu_r5 = 3 * self.mu / r**5

        df1dx = mu_r5 * x**2 - mu_r3
        df1dy = mu_r5 * x * y
        df1dz = mu_r5 * x * z
        df2dy = mu_r5 * y**2 - mu_r3
        df2dz = mu_r5 * y * z
        df3dz = mu_r5 * z**2 - mu_r3

        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [df1dx, df1dy, df1dz, 0, 0, 0],
                [df1dy, df2dy, df2dz, 0, 0, 0],
                [df1dz, df2dz, df3dz, 0, 0, 0],
            ]
        )
        return A
