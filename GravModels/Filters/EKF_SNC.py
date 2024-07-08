import numpy as np
from scipy.integrate import solve_ivp


class ExtendedKalmanFilterSNC:
    def __init__(self, dynamicalModel, measurementModel, Q, R, x0, P0, SNC_flag=True):
        """
        Initializes the Extended Kalman Filter with SNC (Sequential Nonlinear Correction) algorithm.

        Args:
            dynamicalModel: The dynamical model used for state prediction.
            measurementModel: The measurement model used for state update.
            Q: The process noise covariance matrix.
            R: The measurement noise covariance matrix.
            x0: The initial state estimate.
            P0: The initial state covariance matrix.
            SNC_flag: A flag indicating whether to use SNC algorithm (default is True).
        """
        self.dynamicalModel = dynamicalModel
        self.measurementModel = measurementModel
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.SNC_flag = SNC_flag
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
        Q_snc = self._compute_process_noise_covariance(dt)
        self._predict_covariance(Q_snc)

    def update(self, z):
        """
        Updates the state estimate and covariance estimate using the given measurement.

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
            dt (float): Time interval for integration.

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
        Compute the process noise covariance matrix.

        Parameters:
        - dt (float): Time step.

        Returns:
        - Q_snc (numpy.ndarray): Process noise covariance matrix.
        """
        if self.SNC_flag:
            Q_snc = np.vstack(
                (
                    np.hstack((dt**3 / 3 * self.Q, dt**2 / 2 * self.Q)),
                    np.hstack((dt**2 / 2 * self.Q, dt * self.Q)),
                )
            )
        else:
            Q_snc = np.zeros((self.n, self.n))
        return Q_snc

    def _predict_covariance(self, Q_snc):
        """
        Predicts the covariance matrix of the state estimation error.

        Parameters:
        - Q_snc: numpy.ndarray
            The process noise covariance matrix.

        Returns:
        None
        """
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + Q_snc

    def _compute_measurement_jacobian(self):
        """
        Computes the measurement Jacobian matrix.

        This method calculates and returns the measurement Jacobian matrix based on the current state vector.

        Returns:
            numpy.ndarray: The measurement Jacobian matrix.
        """
        return self.measurementModel.jacobian(self.x[:3], self.x[3:6])

    def _compute_kalman_gain(self, H):
        """
        Computes the Kalman gain for the Extended Kalman Filter (EKF).

        Parameters:
        - H: numpy.ndarray
            The measurement matrix.

        Returns:
        - K: numpy.ndarray
            The Kalman gain matrix.
        """
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        return K

    def _update_state_estimate(self, z, K):
        """
        Updates the state estimate using the measurement `z` and the Kalman gain `K`.

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
        - K: Kalman gain matrix
        - H: Measurement matrix

        Returns:
        None
        """
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry
