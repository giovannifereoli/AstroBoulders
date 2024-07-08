import numpy as np


class LinearizedKalmanFilter:
    def __init__(self, Q, R, dx0, P0):
        """
        Initializes the Linear Kalman Filter.

        Args:
            Q (numpy.ndarray): Process noise covariance.
            R (numpy.ndarray): Measurement noise covariance.
            dx0 (numpy.ndarray): Initial state deviation estimate.
            P0 (numpy.ndarray): Initial covariance estimate.
        """
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.dx = dx0  # Initial state deviation estimate
        self.P = P0  # Initial covariance estimate

    def predict(self, F):
        """
        Predicts the state and covariance of the system using the given transition matrix.

        Parameters:
        - F: The transition matrix representing the dynamics of the system.

        Returns:
        None
        """
        self._predict_state(F)
        self._predict_covariance(F)

    def update(self, H, dz):
        """
        Updates the state estimate and covariance estimate using the Kalman filter.

        Parameters:
        - H: The measurement matrix.
        - dz: The measurement residual.

        Returns:
        None
        """
        K = self._compute_kalman_gain(H)
        self._update_state_estimate(H, dz, K)
        self._update_covariance_estimate(H, K)

    def _predict_state(self, F):
        """
        Predicts the state vector using the given transition matrix.

        Args:
            F (numpy.ndarray): The transition matrix.

        Returns:
            None
        """
        self.dx = np.dot(F, self.dx)

    def _predict_covariance(self, F):
        """
        Predicts the covariance matrix of the state estimate using the given state transition matrix.

        Parameters:
            F (numpy.ndarray): State transition matrix.

        Returns:
            None
        """
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def _compute_kalman_gain(self, H):
        """
        Computes the Kalman gain for the given measurement matrix H.

        Parameters:
        - H: Measurement matrix

        Returns:
        - K: Kalman gain
        """
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        return K

    def _update_state_estimate(self, H, dz, K):
        """
        Update the state estimate using the Kalman gain.

        Parameters:
        - H: numpy.ndarray
            The measurement matrix.
        - dz: numpy.ndarray
            The measurement vector.
        - K: numpy.ndarray
            The Kalman gain.

        Returns:
        None
        """
        dy = dz - np.dot(H, self.dx)
        self.dx = self.dx + np.dot(K, dy)

    def _update_covariance_estimate(self, H, K):
        """
        Update the covariance estimate using the Kalman gain.

        Parameters:
        - H: Measurement matrix
        - K: Kalman gain

        Returns:
        None
        """
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry
