import numpy as np

# TODO: add number of pred/corr steps for measurement model


class MinimumVarianceBatchFilter:
    def __init__(self, Q, R, x0, P0):
        """
        Initializes the Minimum Variance Batch Filter.

        Args:
            Q (numpy.ndarray): Process noise covariance.
            R (numpy.ndarray): Measurement noise covariance.
            x0 (numpy.ndarray): Initial state estimate.
            P0 (numpy.ndarray): Initial covariance estimate.
        """
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x0 = x0  # Initial state estimate
        self.P0 = P0  # Initial covariance estimate
        self.x = None  # To be set after batch processing
        self.P = None  # To be set after batch processing

    def batch_process(self, F_list, H_list, z_list):
        """
        Performs batch processing to estimate the state using all measurements.

        Args:
            F_list (list of numpy.ndarray): List of state transition matrices.
            H_list (list of numpy.ndarray): List of measurement matrices.
            z_list (list of numpy.ndarray): List of measurements.

        Returns:
            None
        """
        n = self.x0.shape[0]  # State dimension
        m = z_list[0].shape[0]  # Measurement dimension
        N = len(z_list)  # Number of measurements

        # Initialize batch matrices
        H_batch = np.zeros((N * m, n))
        z_batch = np.zeros(N * m)
        R_batch = np.zeros((N * m, N * m))
        F_batch = np.eye(n)
        Q_batch = np.zeros((n, n))

        # Populate batch matrices
        for i in range(N):
            F_batch = np.dot(F_list[i], F_batch)
            Q_batch = np.dot(F_list[i], np.dot(Q_batch, F_list[i].T)) + self.Q

            H_batch[i * m : (i + 1) * m, :] = np.dot(H_list[i], F_batch)
            z_batch[i * m : (i + 1) * m] = z_list[i]
            R_batch[i * m : (i + 1) * m, i * m : (i + 1) * m] = self.R

        # Solve the batch least squares problem
        HTR_inv = np.linalg.inv(np.dot(H_batch.T, np.linalg.inv(R_batch)).dot(H_batch))
        x_batch = HTR_inv.dot(H_batch.T).dot(np.linalg.inv(R_batch)).dot(z_batch)

        # Update state and covariance
        self.x = self.x0 + x_batch
        self.P = HTR_inv

    def get_state(self):
        """
        Returns the estimated state after batch processing.

        Returns:
            numpy.ndarray: The estimated state.
        """
        return self.x

    def get_covariance(self):
        """
        Returns the estimated state covariance after batch processing.

        Returns:
            numpy.ndarray: The estimated state covariance.
        """
        return self.P
