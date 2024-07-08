import numpy as np
from scipy.integrate import solve_ivp


class ExtendedKalmanFilterDMC:
    def __init__(
        self, dynamicalModel, measurementModel, Q, R, x0, P0, B=np.eye(3), DMC_flag=True
    ):
        """
        Initializes an instance of the EKF_DMC class.

        Parameters:
        - dynamicalModel: The dynamical model used for state prediction.
        - measurementModel: The measurement model used for state update.
        - Q: The process noise covariance matrix.
        - R: The measurement noise covariance matrix.
        - x0: The initial state vector.
        - P0: The initial state covariance matrix.
        - B: The control input matrix (default is identity matrix).
        - DMC_flag: A flag indicating whether to use DMC (default is True).

        Returns:
        - None
        """
        self.dynamicalModel = dynamicalModel
        self.measurementModel = measurementModel
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.B = B
        self.DMC_flag = DMC_flag
        self.n = len(x0)

    def predict(self, dt):
        """
        Predicts the state and covariance of the system at the next time step.

        Args:
            dt (float): The time step size.

        Returns:
            None
        """
        self._integrate_dynamics_and_stm(dt)
        Q_dmc = self._compute_process_noise_covariance(dt)
        self._predict_covariance(Q_dmc)

    def update(self, z):
        """
        Updates the state estimate and covariance estimate based on the given measurement.

        Parameters:
            z (numpy.ndarray): The measurement vector.

        Returns:
            None
        """
        H = self._compute_measurement_jacobian()
        K = self._compute_kalman_gain(H)
        self._update_state_estimate(z, K)
        self._update_covariance_estimate(K, H)

    def _integrate_dynamics_and_stm(self, dt):
        """
        Integrate the dynamics and state transition matrix (STM) over a given time interval.

        Parameters:
        - dt: The time interval over which to integrate the dynamics and STM.

        Returns:
        None
        """
        sol = solve_ivp(
            self.dynamicalModel.dynamics_stm,
            [0, dt],
            np.concatenate((self.x, np.reshape(np.eye(self.n), (self.n**2,)))),
            method="LSODA",
            rtol=2.5e-14,
            atol=2.5e-14,
            t_eval=[dt],
        )
        self.x = sol.y[: self.n, -1]
        self.F = np.reshape(sol.y[self.n :, -1], (self.n, self.n))

    def _compute_process_noise_covariance(self, dt):
        """
        Computes the process noise covariance matrix.

        Parameters:
        - dt (float): Time step between measurements.

        Returns:
        - numpy.ndarray: The process noise covariance matrix.
        """
        if self.DMC_flag:
            return self.calculate_Q_dmc(dt)
        else:
            return np.zeros((self.n, self.n))

    def _predict_covariance(self, Q_dmc):
        """
        Predicts the covariance matrix of the state estimation using the process noise covariance matrix.

        Parameters:
        - Q_dmc: numpy.ndarray
            The process noise covariance matrix.

        Returns:
        None
        """
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + Q_dmc

    def _compute_measurement_jacobian(self):
        """
        Computes the measurement Jacobian matrix.

        Returns:
            The measurement Jacobian matrix.
        """
        return self.measurementModel.jacobian(self.x[:3], self.x[3:6])

    def _compute_kalman_gain(self, H):
        """
        Compute the Kalman gain for the Extended Kalman Filter.

        Parameters:
            H (numpy.ndarray): The measurement matrix.

        Returns:
            numpy.ndarray: The computed Kalman gain.
        """
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        return K

    def _update_state_estimate(self, z, K):
        """
        Updates the state estimate using the measurement z and the Kalman gain K.

        Parameters:
        - z: The measurement vector.
        - K: The Kalman gain matrix.

        Returns:
        None
        """
        y = z - self.measurementModel.get_measurements(
            self.x[:3], self.x[3:6], np.sqrt(self.R[0, 0]), np.sqrt(self.R[1, 1])
        )
        self.x = self.x + np.dot(K, y)

    def _update_covariance_estimate(self, K, H):
        """
        Update the covariance estimate using the Kalman gain and measurement matrix.

        Parameters:
        - K (numpy.ndarray): Kalman gain matrix.
        - H (numpy.ndarray): Measurement matrix.

        Returns:
        None
        """
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry

    def calculate_Q_dmc(self, dt):
        """
        Calculates the process noise covariance matrix Q_dmc for the Discrete-time Markov Chain (DMC) Extended Kalman Filter (EKF).

        Parameters:
        - dt (float): The time step.

        Returns:
        - Q_dmc (numpy.ndarray): The process noise covariance matrix.

        """
        Q_dmc = np.zeros((self.n, self.n))

        for i in range(3):
            dt2 = dt**2
            dt3 = dt**3
            exp_beta_dt = np.exp(-self.B[i, i] * dt)
            exp_2beta_dt = np.exp(-2 * self.B[i, i] * dt)

            Qrr = self.Q[i, i] * (
                1 / (3 * self.B[i, i] ** 2) * dt3
                - 1 / (self.B[i, i] ** 3) * dt2
                + 1 / (self.B[i, i] ** 4) * dt
                - 2 / (self.B[i, i] ** 4) * dt * exp_beta_dt
                + 1 / (2 * self.B[i, i] ** 5) * (1 - exp_2beta_dt)
            )
            Qrv = self.Q[i, i] * (
                1 / (2 * self.B[i, i] ** 2) * dt2
                - 1 / (self.B[i, i] ** 3) * dt
                + 1 / (self.B[i, i] ** 3) * exp_beta_dt * dt
                + 1 / (self.B[i, i] ** 4) * (1 - exp_beta_dt)
                - 1 / (2 * self.B[i, i] ** 4) * (1 - exp_2beta_dt)
            )
            Qrw = self.Q[i, i] * (
                1 / (2 * self.B[i, i] ** 3) * (1 - exp_2beta_dt)
                - 1 / (self.B[i, i] ** 2) * exp_beta_dt * dt
            )
            Qvv = self.Q[i, i] * (
                1 / (self.B[i, i] ** 2) * dt
                - 2 / (self.B[i, i] ** 3) * (1 - exp_beta_dt)
                + 1 / (2 * self.B[i, i] ** 3) * (1 - exp_2beta_dt)
            )
            Qvw = self.Q[i, i] * (
                1 / (2 * self.B[i, i] ** 2) * (1 + exp_2beta_dt)
                - 1 / (self.B[i, i] ** 2) * exp_beta_dt
            )
            Qww = self.Q[i, i] * (1 / (2 * self.B[i, i]) * (1 - exp_2beta_dt))

            Q_dmc[i::3, i::3] = np.array(
                [[Qrr, Qrv, Qrw], [Qrv, Qvv, Qvw], [Qrw, Qvw, Qww]]
            )

        return Q_dmc
