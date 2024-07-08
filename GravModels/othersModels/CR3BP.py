import numpy as np


class CR3BP:
    def __init__(self, mu=0.012150583925359):
        """
        Initialize the CR3BP class.

        Parameters:
        - mu (float): The gravitational parameter of the CR3BP system. Default value is 0.012150583925359.

        Returns:
        None
        """
        self.mu = mu

    def _compute_distances(self, x, y, z):
        """
        Compute the distances from a point (x, y, z) to the two primary bodies in the CR3BP system.

        Parameters:
        - x (float): The x-coordinate of the point.
        - y (float): The y-coordinate of the point.
        - z (float): The z-coordinate of the point.

        Returns:
        - r1 (float): The distance from the point to the first primary body.
        - r2 (float): The distance from the point to the second primary body.
        """
        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - (1 - self.mu)) ** 2 + y**2 + z**2)
        return r1, r2

    def _compute_acceleration(self, x, y, z, vx, vy, vz, r1, r2):
        """
        Compute the acceleration at a given position in the CR3BP model.

        Parameters:
        - x: The x-coordinate of the position.
        - y: The y-coordinate of the position.
        - z: The z-coordinate of the position.
        - vx: The x-component of the velocity.
        - vy: The y-component of the velocity.
        - vz: The z-component of the velocity.
        - r1: The distance from the primary mass.
        - r2: The distance from the secondary mass.

        Returns:
        - ax: The x-component of the acceleration.
        - ay: The y-component of the acceleration.
        - az: The z-component of the acceleration.
        """
        ax = (
            2 * vy
            + x
            - (1 - self.mu) * (x + self.mu) / r1**3
            - self.mu * (x - (1 - self.mu)) / r2**3
        )
        ay = -2 * vx + y - (1 - self.mu) * y / r1**3 - self.mu * y / r2**3
        az = -(1 - self.mu) * z / r1**3 - self.mu * z / r2**3
        return ax, ay, az

    def dynamics(self, t, y):
        """
        Computes the dynamics of the system at a given time and state.

        Parameters:
        t (float): The current time.
        y (list): The current state vector [x, y, z, vx, vy, vz].

        Returns:
        list: The derivative of the state vector [vx, vy, vz, ax, ay, az].
        """
        x, y, z, vx, vy, vz = y
        r1, r2 = self._compute_distances(x, y, z)
        ax, ay, az = self._compute_acceleration(x, y, z, vx, vy, vz, r1, r2)
        return [vx, vy, vz, ax, ay, az]

    def dynamics_stm(self, t, y):
        """
        Computes the dynamics of the state transition matrix (STM) along with the state vector.

        Args:
            t (float): The current time.
            y (list): The state vector containing the position and velocity components along with the STM.

        Returns:
            list: The derivative of the state vector along with the derivative of the STM, flattened into a list.
        """
        stm = np.reshape(y[6:], (6, 6))
        x, y, z, vx, vy, vz = y[:6]

        r1, r2 = self._compute_distances(x, y, z)
        ax, ay, az = self._compute_acceleration(x, y, z, vx, vy, vz, r1, r2)

        A = self.jacobian(t, np.array([x, y, z, vx, vy, vz]))
        d_stm = np.dot(A, stm)

        return [vx, vy, vz, ax, ay, az] + d_stm.flatten().tolist()

    def jacobian(self, t, y):
        """
        Compute the Jacobian matrix for the CR3BP system.

        Parameters:
        - t: Current time
        - y: State vector containing the position and velocity components (x, y, z, vx, vy, vz)

        Returns:
        - A: Jacobian matrix

        The Jacobian matrix is computed based on the given state vector and the current time.
        It represents the linearization of the system dynamics around the given state.

        The Jacobian matrix has the following structure:

        [[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [df1dx, df1dy, df1dz, 0, 2, 0],
         [df1dy, df2dy, df2dz, -2, 0, 0],
         [df1dz, df2dz, df3dz, 0, 0, 0]]

        Note: The values of df1dx, df1dy, df1dz, df2dy, df2dz, and df3dz are computed based on the current state.

        """
        x, y, z, vx, vy, vz = y
        r1, r2 = self._compute_distances(x, y, z)

        df1dx = (
            1
            - (1 - self.mu) / r1**3
            + 3 * (1 - self.mu) * (x + self.mu) ** 2 / r1**5
            - self.mu / r2**3
            + 3 * self.mu * (x + self.mu - 1) ** 2 / r2**5
        )
        df1dy = (
            3 * (1 - self.mu) * (x + self.mu) * y / r1**5
            + 3 * self.mu * (x + self.mu - 1) * y / r2**5
        )
        df1dz = (
            3 * (1 - self.mu) * (x + self.mu) * z / r1**5
            + 3 * self.mu * (x + self.mu - 1) * z / r2**5
        )
        df2dy = (
            1
            - (1 - self.mu) / r1**3
            + 3 * (1 - self.mu) * y**2 / r1**5
            - self.mu / r2**3
            + 3 * self.mu * y**2 / r2**5
        )
        df2dz = 3 * (1 - self.mu) * y * z / r1**5 + 3 * self.mu * y * z / r2**5
        df3dz = (
            -(1 - self.mu) / r1**3
            + 3 * (1 - self.mu) * z**2 / r1**5
            - self.mu / r2**3
            + 3 * self.mu * z**2 / r2**5
        )

        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [df1dx, df1dy, df1dz, 0, 2, 0],
                [df1dy, df2dy, df2dz, -2, 0, 0],
                [df1dz, df2dz, df3dz, 0, 0, 0],
            ]
        )
        return A
