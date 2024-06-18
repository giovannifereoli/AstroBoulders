import os
import numpy as np
import trimesh
from numba import jit, njit, prange


# genera 1 massa per ciascun tetraedro del poliedro
def mass_position(asteroid):
    density = asteroid.density
    obj_file = asteroid.model
    _, file_extension = os.path.splitext(obj_file)
    mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
    scaleFactor = 1e3
    r0 = np.array([0, 0, 0])
    rm_vec = np.zeros((len(mesh.faces), 3))
    mass_vec = np.zeros(len(mesh.faces))
    for i in prange(len(mesh.faces)):
        r1 = mesh.vertices[int(mesh.faces[i][0])] * scaleFactor
        r2 = mesh.vertices[int(mesh.faces[i][1])] * scaleFactor
        r3 = mesh.vertices[int(mesh.faces[i][2])] * scaleFactor
        rm = (r0 + r1 + r2 + r3) / 4
        rm_vec[i][:] = rm
        mass = np.dot(r1 - r0, np.cross(r2 - r0, r3 - r0)) / 6 * density
        mass_vec[i] = mass
    return mass_vec, rm_vec


# genera tre masse per ciascun tetraedro del poliedro
def mass_position_multiple(asteroid):
    density = asteroid.density
    obj_file = asteroid.model
    _, file_extension = os.path.splitext(obj_file)
    mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
    scaleFactor = 1e3
    r0 = np.array([0, 0, 0])
    rm_vec = np.zeros((len(mesh.faces) * 3, 3))
    mass_vec = np.zeros(len(mesh.faces) * 3)
    for i in prange(len(mesh.faces)):
        r1 = mesh.vertices[int(mesh.faces[i][0])] * scaleFactor
        r2 = mesh.vertices[int(mesh.faces[i][1])] * scaleFactor
        r3 = mesh.vertices[int(mesh.faces[i][2])] * scaleFactor

        r1_1 = r1 / 3
        r1_2 = r1 / 3 * 2
        r2_1 = r2 / 3
        r2_2 = r2 / 3 * 2
        r3_1 = r3 / 3
        r3_2 = r3 / 3 * 2

        rm_1 = (r0 + r1_1 + r2_1 + r3_1) / 4
        rm_2 = (r0 + r1_2 + r2_2 + r3_2) / 4
        rm_3 = (r0 + r1 + r2 + r3) / 4
        mass_1 = np.dot(r1_1 - r0, np.cross(r2_1 - r0, r3_1 - r0)) / 6 * density
        mass_2 = np.dot(r1_2 - r0, np.cross(r2_2 - r0, r3_2 - r0)) / 6 * density
        mass_3 = np.dot(r1 - r0, np.cross(r2 - r0, r3 - r0)) / 6 * density
        rm_vec[i * 3][:] = rm_1
        rm_vec[i * 3 + 1][:] = (rm_2 * mass_2 - rm_1 * mass_1) / (mass_2 - mass_1)
        rm_vec[i * 3 + 2][:] = (rm_3 * mass_3 - rm_2 * mass_2) / (mass_3 - mass_2)
        mass_vec[i * 3] = mass_1
        mass_vec[i * 3 + 1] = mass_2 - mass_1
        mass_vec[i * 3 + 2] = mass_3 - mass_2
    return mass_vec, rm_vec


# genera n masse per ciascun tetraedro del poliedro, è il più generale. puoi cambiare l'asteroide e
# renderlo universale se preferisci. Basta avere come input l'asteroide di riferimento
def mass_position_multiple_choose(asteroid, number_of_intervals):
    density = asteroid.density
    obj_file = asteroid.model
    _, file_extension = os.path.splitext(obj_file)
    mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
    scaleFactor = 1e3
    r0 = np.array([0, 0, 0])
    rm_vec = np.zeros((len(mesh.faces) * number_of_intervals, 3))
    mass_vec = np.zeros(len(mesh.faces) * number_of_intervals)
    for i in prange(len(mesh.faces)):
        r1 = mesh.vertices[int(mesh.faces[i][0])] * scaleFactor
        r2 = mesh.vertices[int(mesh.faces[i][1])] * scaleFactor
        r3 = mesh.vertices[int(mesh.faces[i][2])] * scaleFactor

        r1_vec = np.zeros((int(number_of_intervals) - 1, 3))
        r2_vec = np.zeros((int(number_of_intervals) - 1, 3))
        r3_vec = np.zeros((int(number_of_intervals) - 1, 3))
        for j in range(len(r1_vec)):
            r1_vec[j] = r1 / number_of_intervals * int(j + 1)
            r2_vec[j] = r2 / number_of_intervals * int(j + 1)
            r3_vec[j] = r3 / number_of_intervals * int(j + 1)
        r1_vec = np.append(r1_vec, [r1], axis=0)
        r2_vec = np.append(r2_vec, [r2], axis=0)
        r3_vec = np.append(r3_vec, [r3], axis=0)

        rm_pos = np.zeros((int(number_of_intervals), 3))
        for j in range(int(number_of_intervals)):
            rm_pos[j] = (r0 + r1_vec[j] + r2_vec[j] + r3_vec[j]) / 4

        mass_value = np.zeros(int(number_of_intervals))
        for j in range(int(number_of_intervals)):
            mass_value[j] = (
                np.dot(r1_vec[j] - r0, np.cross(r2_vec[j] - r0, r3_vec[j] - r0))
                / 6
                * density
            )

        rm_vec[i * number_of_intervals] = rm_pos[0]
        for j in range(1, int(number_of_intervals)):
            rm_vec[i * number_of_intervals + j] = (
                rm_pos[j] * mass_value[j] - rm_pos[j - 1] * mass_value[j - 1]
            ) / (mass_value[j] - mass_value[j - 1])

        mass_vec[i * number_of_intervals] = mass_value[0]
        for j in range(1, int(number_of_intervals)):
            mass_vec[i * number_of_intervals + j] = mass_value[j] - mass_value[j - 1]
    return mass_vec, rm_vec


# calcolo potenziale di didymos
@njit(cache=True, parallel=True)
def mascon_potential(position, mass_vector, rm_vector):
    U = 0
    G = 6.67408 * 10**-11
    for i in prange(len(mass_vector)):
        radius = np.sqrt(
            (position[0][0] - rm_vector[i][0]) ** 2
            + (position[0][1] - rm_vector[i][1]) ** 2
            + (position[0][2] - rm_vector[i][2]) ** 2
        )
        U -= G * mass_vector[i] / radius
    return U


# calcolo accelerazione di didymos
@njit(cache=True, parallel=True)
def mascon_acceleration(position, mass_vector, rm_vector):
    acc = np.zeros(3)
    G = 6.67408 * 10**-11
    for i in prange(len(mass_vector)):
        radius = np.sqrt(
            (position[0][0] - rm_vector[i][0]) ** 2
            + (position[0][1] - rm_vector[i][1]) ** 2
            + (position[0][2] - rm_vector[i][2]) ** 2
        )
        acc += (
            G
            * mass_vector[i]
            / radius**3
            * np.array(
                [
                    position[0][0] - rm_vector[i][0],
                    position[0][1] - rm_vector[i][1],
                    position[0][2] - rm_vector[i][2],
                ]
            )
        )
    return -acc
