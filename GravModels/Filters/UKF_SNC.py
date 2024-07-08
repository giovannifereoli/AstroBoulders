import numpy as np
from scipy.integrate import solve_ivp


class UnscentedKalmanFilterSNC:
    def __init__(
        self,
        dynamicalModel,
        measurementModel,
        Q,
        R,
        x0,
        P0,
        alpha=1e-3,
        beta=2,
        kappa=0,
        SNC_flag=True,
    ):
        """
        Initializes the Unscented Kalman Filter with SNC (State Noise Compensation) algorithm.

        Args:
            dynamicalModel: The dynamical model used for state prediction.
            measurementModel: The measurement model used for state update.
            Q: The process noise covariance matrix.
            R: The measurement noise covariance matrix.
            x0: The initial state estimate.
            P0: The initial state covariance matrix.
            alpha: Scaling parameter for the sigma points.
            beta: Parameter to incorporate prior knowledge of the distribution (2 is optimal for Gaussian distributions).
            kappa: Secondary scaling parameter.
            SNC_flag: A flag indicating whether to use the SNC algorithm (default is True).
        """
        self.dynamicalModel = dynamicalModel
        self.measurementModel = measurementModel
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.SNC_flag = SNC_flag
        self.n = len(x0)

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)

        # Weights for means and covariance
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)

    def predict(self, dt):
        """
        Predicts the state and covariance of the system at the next time step.

        Args:
            dt (float): The time step size.

        Returns:
            None
        """
        sigma_points = self._generate_sigma_points(self.x, self.P)
        predicted_sigma_points = np.array(
            [self.dynamics_function(sp, dt) for sp in sigma_points]
        )

        self.x = np.dot(self.Wm, predicted_sigma_points)
        self.P = self.Q.copy()
        for i in range(2 * self.n + 1):
            y = predicted_sigma_points[i] - self.x
            self.P += self.Wc[i] * np.outer(y, y)

        if self.SNC_flag:
            Q_snc = self._compute_process_noise_covariance(dt)
            self.P += Q_snc

    def update(self, z):
        """
        Updates the state estimate and covariance estimate using the given measurement.

        Parameters:
            z (numpy.ndarray): The measurement vector.

        Returns:
            None
        """
        sigma_points = self._generate_sigma_points(self.x, self.P)
        predicted_measurements = np.array(
            [self.measurement_function(sp) for sp in sigma_points]
        )

        z_mean = np.dot(self.Wm, predicted_measurements)
        S = self.R.copy()
        for i in range(2 * self.n + 1):
            y = predicted_measurements[i] - z_mean
            S += self.Wc[i] * np.outer(y, y)

        cross_covariance = np.zeros((self.n, len(z)))
        for i in range(2 * self.n + 1):
            cross_covariance += self.Wc[i] * np.outer(
                sigma_points[i] - self.x, predicted_measurements[i] - z_mean
            )

        K = np.dot(cross_covariance, np.linalg.inv(S))
        self.x += np.dot(K, z - z_mean)
        self.P -= np.dot(K, np.dot(S, K.T))

    def _generate_sigma_points(self, x, P):
        """
        Generates sigma points for the Unscented Kalman Filter.

        Parameters:
            x (numpy.ndarray): The state vector.
            P (numpy.ndarray): The covariance matrix.

        Returns:
            numpy.ndarray: The generated sigma points.
        """
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = x
        sqrt_P = np.linalg.cholesky((self.n + self.lambda_) * P)

        for i in range(self.n):
            sigma_points[i + 1] = x + sqrt_P[:, i]
            sigma_points[self.n + i + 1] = x - sqrt_P[:, i]

        return sigma_points

    def _compute_process_noise_covariance(self, dt):
        """
        Compute the process noise covariance matrix with SNC.

        Parameters:
        - dt (float): Time step.

        Returns:
        - Q_snc (numpy.ndarray): Process noise covariance matrix.
        """
        Q_snc = np.vstack(
            (
                np.hstack((dt**3 / 3 * self.Q, dt**2 / 2 * self.Q)),
                np.hstack((dt**2 / 2 * self.Q, dt * self.Q)),
            )
        )
        return Q_snc

    def dynamics_function(self, x, dt):
        """
        Dynamics function for the UKF prediction step.

        Parameters:
            x (numpy.ndarray): The state vector.
            dt (float): The time step.

        Returns:
            numpy.ndarray: The predicted state vector.
        """
        sol = solve_ivp(
            self.dynamicalModel.dynamics,
            [0, dt],
            x,
            method="LSODA",
            rtol=2.5e-14,
            atol=2.5e-14,
            t_eval=[dt],
        )
        return sol.y[:, -1]

    def measurement_function(self, x):
        """
        Measurement function for the UKF update step.

        Parameters:
            x (numpy.ndarray): The state vector.

        Returns:
            numpy.ndarray: The measurement vector.
        """
        return self.measurementModel.get_measurements(x[:3], x[3:6], 0, 0)
