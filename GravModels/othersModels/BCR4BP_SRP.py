import numpy as np


class BCR4BP_SRP:
    def __init__(
        self,
        mu=0.012150583925359,
        m_star=6.0458e24,  # Kilograms
        l_star=3.844e5,  # Kilometers
        t_star=375200,  # Seconds
        SRP_flag=True,
        sun_mass=1.9885e30,  # Kilograms
        sun_angular_velocity=-9.25195985e-1,  # rad/s
        sun_distance=149.9844e6,  # Kilometers
    ):
        """
        Initializes the BCR4BP class.

        Parameters:
        - mu (float): The gravitational parameter of the Earth-Moon system.
        - m_star (float): Mass of the Earth-Moon system in kilograms.
        - l_star (float): Distance between the Earth and the Moon in kilometers.
        - t_star (float): Time period of the Earth-Moon system in seconds.
        - SRP_flag (bool): Flag indicating whether to consider solar radiation pressure.
        - sun_mass (float): Mass of the Sun in kilograms.
        - sun_angular_velocity (float): Angular velocity of the Sun in rad/s.
        - sun_distance (float): Distance between the Sun and the Earth-Moon system in kilometers.
        """
        self.mu = mu
        self.m_star = m_star
        self.l_star = l_star
        self.t_star = t_star
        self.SRP_flag = SRP_flag
        self.ms = sun_mass / m_star  # Scaled mass of the Sun
        self.ws = sun_angular_velocity  # Scaled angular velocity of the Sun
        self.rho = sun_distance / l_star  # Scaled distance Sun-(Earth+Moon)

    def _compute_crtbp_acceleration(self, x, y, z, r1, r2):
        """
        Compute the acceleration of a particle in the Circular Restricted Three-Body Problem (CRTBP).

        Parameters:
            x (float): x-coordinate of the particle.
            y (float): y-coordinate of the particle.
            z (float): z-coordinate of the particle.
            r1 (float): Distance from the particle to the primary body.
            r2 (float): Distance from the particle to the secondary body.

        Returns:
            numpy.ndarray: Array containing the x, y, and z components of the acceleration.

        """
        ax = (
            2 * y[3]
            + x[0]
            - (1 - self.mu) * (x[0] + self.mu) / r1**3
            - self.mu * (x[0] - (1 - self.mu)) / r2**3
        )
        ay = -2 * x[3] + x[1] - (1 - self.mu) * x[1] / r1**3 - self.mu * x[1] / r2**3
        az = -(1 - self.mu) * x[2] / r1**3 - self.mu * x[2] / r2**3
        return np.array([ax, ay, az])

    def _compute_bcr4bp_acceleration(self, x, y, t):
        """
        Compute the acceleration for the BCR4BP (Bicircular Restricted Three-Body Problem) model.

        Args:
            x (array-like): Position vector [x, y, z].
            y (float): Dummy variable.
            t (float): Time variable.

        Returns:
            tuple: A tuple containing the acceleration vector and the rho vector.
                - acceleration (array-like): Acceleration vector [ax, ay, az].
                - rho_vec (array-like): Rho vector [rho_x, rho_y, rho_z].
        """
        rho_vec = self.rho * np.array([np.cos(self.ws * t), np.sin(self.ws * t), 0])
        r3 = np.sqrt(
            (x[0] - self.rho * np.cos(self.ws * t)) ** 2
            + (x[1] - self.rho * np.sin(self.ws * t)) ** 2
            + x[2] ** 2
        )
        a_4b = np.array(
            [
                -self.ms * (x[0] - self.rho * np.cos(self.ws * t)) / r3**3
                - self.ms * np.cos(self.ws * t) / self.rho**2,
                -self.ms * (x[1] - self.rho * np.sin(self.ws * t)) / r3**3
                - self.ms * np.sin(self.ws * t) / self.rho**2,
                -self.ms * x[2] / r3**3,
            ]
        )
        return a_4b, rho_vec

    def _compute_srp_acceleration(self, rho_vec, m=2000, A=1e-6, Cr=1):
        """
        Compute the acceleration due to Solar Radiation Pressure (SRP).

        Parameters:
        - rho_vec: numpy array representing the position vector of the spacecraft with respect to the Sun.
        - m: spacecraft mass in kilograms (default: 2000 kg).
        - A: spacecraft area in square meters (default: 1e-6 m^2).
        - Cr: spacecraft radiation pressure coefficient (default: 1).

        Returns:
        - a_srp: numpy array representing the acceleration due to SRP.

        Note:
        - The method assumes that the units of self.m_star, self.l_star, and self.t_star are consistent with the SI system.
        - The method also assumes that self.SRP_flag is a boolean indicating whether SRP is enabled or not.
        """
        P = 4.56 / (
            (self.m_star * self.l_star / self.t_star**2) / self.l_star**2
        )  # N x km^-2
        m_scaled = m / self.m_star  # Scaled spacecraft mass
        A_scaled = A / self.l_star**2  # Scaled spacecraft area
        dist_coeff = 1 if self.SRP_flag else 0  # 1AU / r_4
        a_srp = -(Cr * A_scaled * P * dist_coeff / m_scaled) * rho_vec
        return a_srp

    def dynamics(self, t, y):
        """
        Computes the dynamics of the system at a given time and state.

        Parameters:
            t (float): The current time.
            y (list): The state vector containing position and velocity components.

        Returns:
            list: The derivative of the state vector, representing the acceleration components.

        """
        x = y[:3]
        vx, vy, vz = y[3:]
        r1 = np.sqrt((x[0] + self.mu) ** 2 + x[1] ** 2 + x[2] ** 2)
        r2 = np.sqrt((x[0] - (1 - self.mu)) ** 2 + x[1] ** 2 + x[2] ** 2)

        a_crtbp = self._compute_crtbp_acceleration(x, y, r1, r2)
        a_4b, rho_vec = self._compute_bcr4bp_acceleration(x, y, t)
        a_srp = self._compute_srp_acceleration(rho_vec)

        return [vx, vy, vz, *(a_crtbp + a_4b + a_srp)]
