import numpy as np

# TODO: should I add white noise Cu(t) to the dynamics?


class CR3BP_DMC:
    def __init__(self, mu=0.012150583925359, B=np.eye(3)):
        """
        Initialize the CR3BP_DMC class.

        Parameters:
        - mu (float): The gravitational parameter of the CR3BP system. Default is 0.012150583925359.
        - B (ndarray): The transformation matrix B. Default is the identity matrix of shape (3, 3).
        """
        self.mu = mu
        self.B = B

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

    def _compute_crtbp_acceleration(self, x, y, z, vx, vy, vz, r1, r2):
        """
        Computes the acceleration in the Circular Restricted Three-Body Problem (CR3BP) model.

        Parameters:
        - x, y, z: Cartesian coordinates of the spacecraft
        - vx, vy, vz: Velocities of the spacecraft
        - r1, r2: Distances from the spacecraft to the two primary bodies

        Returns:
        - ax, ay, az: Acceleration components in the CR3BP model
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

    def _compute_dmc_dynamics(self, wx, wy, wz):
        """
        Compute the dynamics of the DMC (Dynamical Model Coefficients) system.

        Parameters:
        - wx: Angular velocity around the x-axis.
        - wy: Angular velocity around the y-axis.
        - wz: Angular velocity around the z-axis.

        Returns:
        - d_wx: Rate of change of angular velocity around the x-axis.
        - d_wy: Rate of change of angular velocity around the y-axis.
        - d_wz: Rate of change of angular velocity around the z-axis.
        """
        d_wx = -self.B[0, 0] * wx
        d_wy = -self.B[1, 1] * wy
        d_wz = -self.B[2, 2] * wz
        return d_wx, d_wy, d_wz

    def dynamics_stm(self, t, y):
        """
        Computes the dynamics of the state transition matrix (STM) for the CR3BP DMC model.

        Args:
            t (float): The current time.
            y (list): The state vector containing the current values of the variables.

        Returns:
            list: The updated state vector including the dynamics of the STM.

        """
        stm = np.reshape(y[9:], (9, 9))
        x, y, z, vx, vy, vz, wx, wy, wz = y[:9]

        r1, r2 = self._compute_distances(x, y, z)
        ax, ay, az = self._compute_crtbp_acceleration(x, y, z, vx, vy, vz, r1, r2)
        d_wx, d_wy, d_wz = self._compute_dmc_dynamics(wx, wy, wz)

        A_prime = self.jacobian(t, np.array([x, y, z, vx, vy, vz, wx, wy, wz]))
        d_stm = np.dot(A_prime, stm)

        return [
            vx,
            vy,
            vz,
            ax + wx,
            ay + wy,
            az + wz,
            d_wx,
            d_wy,
            d_wz,
        ] + d_stm.flatten().tolist()

    def jacobian(self, t, y):
        """
        Compute the Jacobian matrix for the CR3BP_DMC model.

        Parameters:
        - t: Current time
        - y: State vector containing the variables x, y, z, vx, vy, vz, wx, wy, wz

        Returns:
        - A_prime: Jacobian matrix of shape (9, 9)
        """

        x, y, z, vx, vy, vz, wx, wy, wz = y
        r1, r2 = self._compute_distances(x, y, z)

        df1dx, df1dy, df1dz, df2dy, df2dz, df3dz = self._compute_variational_equations(
            x, y, z, r1, r2
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

        D = np.block([[np.zeros((3, 3))], [np.eye(3)]])
        A_prime = np.block([[A, D], [np.zeros((3, 6)), -self.B]])

        return A_prime

    def _compute_variational_equations(self, x, y, z, r1, r2):
        """
        Compute the variational equations for the CR3BP model.

        This method calculates the partial derivatives of the equations of motion with respect to the state variables.
        The equations are used to study the stability and sensitivity of the system.

        Parameters:
        - x (float): The x-coordinate of the position vector.
        - y (float): The y-coordinate of the position vector.
        - z (float): The z-coordinate of the position vector.
        - r1 (float): The distance from the primary mass.
        - r2 (float): The distance from the secondary mass.

        Returns:
        - df1dx (float): The partial derivative of the first equation with respect to x.
        - df1dy (float): The partial derivative of the first equation with respect to y.
        - df1dz (float): The partial derivative of the first equation with respect to z.
        - df2dy (float): The partial derivative of the second equation with respect to y.
        - df2dz (float): The partial derivative of the second equation with respect to z.
        - df3dz (float): The partial derivative of the third equation with respect to z.
        """
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
        return df1dx, df1dy, df1dz, df2dy, df2dz, df3dz
