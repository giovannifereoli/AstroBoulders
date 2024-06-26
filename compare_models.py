import numpy as np
import os
import trimesh
from scipy.spatial import ConvexHull
from GravModels.Models.Mascon import Mascon
from GravModels.Models.Polyhedral import Polyhedral
from GravModels.Models.Pines import Pines
from GravModels.CelestialBodies.Asteroids import Itokawa
from GravModels.utils.Plotting import plot_acceleration_contours
from GravModels.utils.Utils import compute_acceleration_grid


def load_mesh_vertices(asteroid):
    obj_file = asteroid.model
    _, file_extension = os.path.splitext(obj_file)
    mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
    return mesh.vertices * 1000


def compute_accelerations(asteroid, feature_x, feature_y, feature_z, num):
    poly_model = Polyhedral(asteroid)
    acc_vec_poly_x, acc_vec_poly_y, acc_vec_poly_z = compute_acceleration_grid(
        num,
        feature_x,
        feature_y,
        feature_z,
        lambda point: poly_model.calculate_acceleration(point),
    )

    pines_model = Pines(asteroid)
    acc_vec_spher_x, acc_vec_spher_y, acc_vec_spher_z = compute_acceleration_grid(
        num,
        feature_x,
        feature_y,
        feature_z,
        lambda point: pines_model.calculate_acceleration(point),
    )

    mascon_model = Mascon(asteroid)
    acc_vec_mascon_x, acc_vec_mascon_y, acc_vec_mascon_z = compute_acceleration_grid(
        num,
        feature_x,
        feature_y,
        feature_z,
        lambda point: mascon_model.calculate_acceleration(point),
    )

    return (
        acc_vec_poly_x,
        acc_vec_poly_y,
        acc_vec_poly_z,
        acc_vec_spher_x,
        acc_vec_spher_y,
        acc_vec_spher_z,
        acc_vec_mascon_x,
        acc_vec_mascon_y,
        acc_vec_mascon_z,
    )


def plot_all_contours(
    feature_x,
    feature_y,
    feature_z,
    acc_vec_poly,
    acc_vec_spher,
    acc_vec_mascon,
    vertices,
    hull,
):
    plot_acceleration_contours(
        feature_x,
        feature_y,
        acc_vec_poly[2],
        vertices[:, :2],
        hull[0],
        "Polyhedral xy-plane",
        ["Didymos", "x", "y"],
    )
    plot_acceleration_contours(
        feature_x,
        feature_z,
        acc_vec_poly[1],
        vertices[:, [0, 2]],
        hull[1],
        "Polyhedral xz-plane",
        ["Didymos", "x", "z"],
    )
    plot_acceleration_contours(
        feature_y,
        feature_z,
        acc_vec_poly[0],
        vertices[:, [1, 2]],
        hull[2],
        "Polyhedral yz-plane",
        ["Didymos", "y", "z"],
    )

    plot_acceleration_contours(
        feature_x,
        feature_y,
        acc_vec_spher[2],
        vertices[:, :2],
        hull[0],
        "Spherical Harmonics xy-plane",
        ["Didymos", "x", "y"],
    )
    plot_acceleration_contours(
        feature_x,
        feature_z,
        acc_vec_spher[1],
        vertices[:, [0, 2]],
        hull[1],
        "Spherical Harmonics xz-plane",
        ["Didymos", "x", "z"],
    )
    plot_acceleration_contours(
        feature_y,
        feature_z,
        acc_vec_spher[0],
        vertices[:, [1, 2]],
        hull[2],
        "Spherical Harmonics yz-plane",
        ["Didymos", "y", "z"],
    )

    plot_acceleration_contours(
        feature_x,
        feature_y,
        acc_vec_mascon[2],
        vertices[:, :2],
        hull[0],
        "Mascon xy-plane",
        ["Didymos", "x", "y"],
    )
    plot_acceleration_contours(
        feature_x,
        feature_z,
        acc_vec_mascon[1],
        vertices[:, [0, 2]],
        hull[1],
        "Mascon xz-plane",
        ["Didymos", "x", "z"],
    )
    plot_acceleration_contours(
        feature_y,
        feature_z,
        acc_vec_mascon[0],
        vertices[:, [1, 2]],
        hull[2],
        "Mascon yz-plane",
        ["Didymos", "y", "z"],
    )


def main():
    asteroid = Itokawa()
    num = 10

    feature_x = np.linspace(-2000, 2000, num)
    feature_y = np.linspace(-2000, 2000, num)
    feature_z = np.linspace(-2000, 2000, num)

    acc_vec = compute_accelerations(asteroid, feature_x, feature_y, feature_z, num)
    acc_vec_poly = acc_vec[:3]
    acc_vec_spher = acc_vec[3:6]
    acc_vec_mascon = acc_vec[6:]

    vertices = load_mesh_vertices(asteroid)
    xy_vertices = vertices[:, :2]
    xz_vertices = vertices[:, [0, 2]]
    yz_vertices = vertices[:, [1, 2]]

    hull_xy = ConvexHull(xy_vertices)
    hull_xz = ConvexHull(xz_vertices)
    hull_yz = ConvexHull(yz_vertices)

    plot_all_contours(
        feature_x,
        feature_y,
        feature_z,
        acc_vec_poly,
        acc_vec_spher,
        acc_vec_mascon,
        vertices,
        (hull_xy, hull_xz, hull_yz),
    )


if __name__ == "__main__":
    main()
