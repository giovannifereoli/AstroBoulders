import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import trimesh


def set_axes_equal(ax):
    """
    Set the axes of a 3D plot to have equal scales.

    Parameters:
    - ax: The 3D axes object.

    Returns:
    None
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_acceleration_contours(
    feature_x, feature_y, acc_vec, vertices, hull, title, plane_labels
):
    """
    Plot acceleration contours.

    Args:
      feature_x (array-like): X-axis values for the contour plot.
      feature_y (array-like): Y-axis values for the contour plot.
      acc_vec (array-like): Acceleration vectors.
      vertices (array-like): Vertices of the hull.
      hull (object): Convex hull object.
      title (str): Title of the plot.
      plane_labels (list): Labels for the X and Y axes.

    Returns:
      None
    """
    X, Y = np.meshgrid(feature_x, feature_y)
    plt.contourf(X, Y, np.transpose(np.linalg.norm(acc_vec, axis=2)), 100)
    plt.fill(vertices[hull.vertices, 0], vertices[hull.vertices, 1], facecolor="gray")
    plt.title(title)
    plt.xlabel(f"{plane_labels[1]} [m]")
    plt.ylabel(f"{plane_labels[2]} [m]")
    plt.colorbar()
    plt.axis("equal")
    plt.axis("tight")
    plt.show()


def plot_asteroid_3d(asteroid, disc=10, ax=None, Trajectory=None):
    """
    Plot a 3D representation of an asteroid.

    Parameters:
    - asteroid: An object representing the asteroid.
    - disc: The level of detail of the mesh. Higher values result in a more detailed mesh.
    - ax: The matplotlib 3D axes to plot on. If None, a new figure and axes will be created.
    - Trajectory: An optional trajectory to plot along with the asteroid.

    Returns:
    None
    """
    obj_file = asteroid.model
    _, file_extension = os.path.splitext(obj_file)
    mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
    vertices = np.array(mesh.vertices * disc) * 100  # TODO: what's that??
    faces = mesh.faces

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Set labels for the axes
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_ylabel("y [m]", fontsize=10)
    ax.set_zlabel("z [m]", fontsize=10)

    mesh = Poly3DCollection(
        [vertices[face] for face in faces],
        facecolor="gray",
        edgecolor="k",
        linewidths=1,
        alpha=1,
    )
    ax.add_collection3d(mesh)

    # Determine plot limits
    if Trajectory is not None:
        x_min = min(vertices[:, 0].min(), Trajectory[:, 0].min())
        x_max = max(vertices[:, 0].max(), Trajectory[:, 0].max())
        y_min = min(vertices[:, 1].min(), Trajectory[:, 1].min())
        y_max = max(vertices[:, 1].max(), Trajectory[:, 1].max())
        z_min = min(vertices[:, 2].min(), Trajectory[:, 2].min())
        z_max = max(vertices[:, 2].max(), Trajectory[:, 2].max())
        ax.plot(
            Trajectory[:, 0],
            Trajectory[:, 1],
            Trajectory[:, 2],
            color="black",
            linewidth=2,
        )
    else:
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)

    # Show the plot
    plt.show()


def plot_range_and_range_rate(t_eval, measurements):
    """
    Plot the range and range rate as subplots.

    Parameters:
    - t_eval: A list of time points at which the measurements were taken.
    - measurements: A list of tuples containing the range and range rate measurements.
    """
    ranges, range_rates = zip(*measurements)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(t_eval, ranges, label="Range")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Range (meters)")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t_eval, range_rates, label="Range Rate", color="orange")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Range Rate (meters/second)")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()


def plot_trajectory_errors(
    t_eval,
    solution_true,
    filtered_state,
    covariance,
    initial_state_true,
    filename="ErrorsPosVel.pdf",
):
    """
    Plots the trajectory errors between the true trajectory and the estimated trajectory.

    Parameters:
    - t_eval (array-like): The time points at which the trajectory is evaluated.
    - solution_true (array-like): The true trajectory.
    - filtered_state (array-like): The estimated trajectory.
    - covariance (array-like): The covariance matrix of the estimated trajectory.
    - initial_state_true (array-like): The initial state of the true trajectory.
    - filename (str): The filename to save the plot (default is "ErrorsPosVel.pdf").

    Returns:
    None
    """
    # Calculate error between true trajectory and estimated trajectory
    error_rv = solution_true.y[:, :-1] - filtered_state[:6, :]

    # Calculate 3-sigma bound for error
    sigma_bound = 3 * np.sqrt(np.abs(np.diagonal(covariance, axis1=0, axis2=1)))

    fig, axs = plt.subplots(2, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=1, hspace=1)

    labels = [
        r"$\varepsilon_x$ [n.d.]",
        r"$\varepsilon_y$ [n.d.]",
        r"$\varepsilon_z$ [n.d.]",
        r"$\varepsilon_{\dot{x}}$ [n.d.]",
        r"$\varepsilon_{\dot{y}}$ [n.d.]",
        r"$\varepsilon_{\dot{z}}$ [n.d.]",
    ]

    for i in range(len(initial_state_true)):
        row_index, col_index = divmod(i, 3)
        axs[row_index, col_index].semilogy(
            t_eval[:-1],
            np.abs(error_rv[i]),
            linestyle="-",
            color="red",
            label=r"$\varepsilon$",
        )
        axs[row_index, col_index].semilogy(
            t_eval[:-1],
            sigma_bound[i],
            linestyle="--",
            color="black",
            label="3$\sigma$",
        )
        axs[row_index, col_index].set_ylabel(labels[i])
        axs[row_index, col_index].set_xlabel(r"t [n.d.]")
        axs[row_index, col_index].legend(loc="lower left")
        axs[row_index, col_index].grid(True, which="both", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.show()


def plot_residual_dynamics_errors(
    t_eval,
    resdyn_true,
    filtered_state,
    covariance,
    initial_state_true,
    filename="ErrorsResDyn.pdf",
):
    """
    Plots the error between true residual dynamics and estimated residual dynamics.

    Parameters:
    - t_eval (array-like): The time values for evaluation.
    - resdyn_true (array-like): The true residual dynamics.
    - filtered_state (array-like): The estimated residual dynamics.
    - covariance (array-like): The covariance matrix.
    - initial_state_true (array-like): The initial true state.
    - filename (str): The filename to save the plot (default is "ErrorsResDyn.pdf").

    Returns:
    None
    """
    # Calculate error between true residual dynamics and estimated residual dynamics
    error_resdyn = resdyn_true.T[3:, :] - filtered_state[6:, :]

    # Calculate 3-sigma bound for error
    sigma_bound = 3 * np.sqrt(np.abs(np.diagonal(covariance, axis1=0, axis2=1)))

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    plt.subplots_adjust(wspace=1, hspace=1)

    labels = [
        r"$\varepsilon_{w_x}$ [n.d.]",
        r"$\varepsilon_{w_y}$ [n.d.]",
        r"$\varepsilon_{w_z}$ [n.d.]",
    ]

    for i in range(3):
        axs[i].semilogy(
            t_eval[:-1],
            np.abs(error_resdyn[i]),
            linestyle="-",
            color="blue",
            label=r"$\varepsilon$",
        )
        axs[i].semilogy(
            t_eval[:-1],
            sigma_bound[len(initial_state_true) + i],
            linestyle="--",
            color="black",
            label="3$\sigma$",
        )
        axs[i].set_ylabel(labels[i])
        axs[i].set_xlabel(r"t [n.d.]")
        axs[i].legend(loc="upper right")
        axs[i].grid(True, which="both", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.show()


def plot_postfit_radiometric_errors(
    t_eval,
    measurements_true,
    filtered_state,
    measurement_model,
    sigma_range,
    sigma_range_rate,
    filename="PostfitRadiometricErrors.pdf",
):
    """
    Plots the postfit radiometric measurement errors for a given set of measurements.

    Parameters:
    - t_eval (array-like): The time values at which the measurements are evaluated.
    - measurements_true (array-like): The true measurements.
    - filtered_state (array-like): The filtered state estimates.
    - measurement_model (object): The measurement model used to calculate estimated measurements.
    - sigma_range (float): The standard deviation of the range measurement errors.
    - sigma_range_rate (float): The standard deviation of the range rate measurement errors.
    - filename (str): The name of the output file to save the plot (default is "PostfitMeasurementErrors.pdf").

    Returns:
    None
    """
    # Calculate postfit errors in measurements
    postfit_errors = []
    for i in range(len(t_eval) - 1):
        estimated_measurement = measurement_model.get_measurements(
            filtered_state[:3, i], filtered_state[3:6, i], 0, 0
        )
        true_measurement = measurements_true[i]
        postfit_error = true_measurement - estimated_measurement
        postfit_errors.append(postfit_error)

    postfit_errors = np.array(postfit_errors).T

    # Calculate 3-sigma bound for measurement errors
    sigma_bound = np.array([3 * sigma_range, 3 * sigma_range_rate])

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)

    labels = [
        r"$\varepsilon_{\text{range}}$ [n.d.]",
        r"$\varepsilon_{\text{range rate}}$ [n.d.]",
    ]

    for i in range(2):
        axs[i].semilogy(
            t_eval[:-1],
            np.abs(postfit_errors[i]),
            linestyle="-",
            color="blue",
            label=r"$\varepsilon$",
        )
        axs[i].semilogy(
            t_eval[:-1],
            sigma_bound[i],
            linestyle="--",
            color="black",
            label="3$\sigma$",
        )
        axs[i].set_ylabel(labels[i])
        axs[i].set_xlabel(r"t [n.d.]")
        axs[i].legend(loc="upper right")
        axs[i].grid(True, which="both", linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.show()
