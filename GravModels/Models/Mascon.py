import os
import numpy as np
import trimesh
from numba import njit, prange


class Mascon:
    def __init__(self, asteroid, number_of_intervals=3):
        """
        Initializes a Mascon object.

        Parameters:
        - asteroid: An instance of the Asteroid class representing the asteroid.
        - number_of_intervals: The number of intervals to use for calculating mass position.

        Returns:
        None
        """
        self.asteroid = asteroid
        self.density = asteroid.density
        self.G = 6.67408 * 10**-11
        self.scaleFactor = 1e3
        self._load_mesh()
        self._calculate_mass_position(number_of_intervals)

    def _load_mesh(self):
        """
        Loads the mesh for the asteroid model.

        This method loads the mesh for the asteroid model specified in the `asteroid` attribute.
        It uses the `trimesh.load_mesh` function to load the mesh from the specified file.

        Args:
            None

        Returns:
            None
        """
        obj_file = self.asteroid.model
        _, file_extension = os.path.splitext(obj_file)
        self.mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])

    def _calculate_mass_position(self, number_of_intervals):
        """
        Calculate the mass and position vectors for each interval of a mesh.

        Args:
            number_of_intervals (int): The number of intervals to divide each face of the mesh.

        Returns:
            None

        """
        mesh = self.mesh
        r0 = np.array([0, 0, 0])
        rm_vec = np.zeros((len(mesh.faces) * number_of_intervals, 3))
        mass_vec = np.zeros(len(mesh.faces) * number_of_intervals)

        for i in prange(len(mesh.faces)):
            r1 = mesh.vertices[int(mesh.faces[i][0])] * self.scaleFactor
            r2 = mesh.vertices[int(mesh.faces[i][1])] * self.scaleFactor
            r3 = mesh.vertices[int(mesh.faces[i][2])] * self.scaleFactor

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
            mass_value = np.zeros(int(number_of_intervals))

            for j in range(int(number_of_intervals)):
                rm_pos[j] = (r0 + r1_vec[j] + r2_vec[j] + r3_vec[j]) / 4
                mass_value[j] = (
                    np.dot(r1_vec[j] - r0, np.cross(r2_vec[j] - r0, r3_vec[j] - r0))
                    / 6
                    * self.density
                )

            rm_vec[i * number_of_intervals] = rm_pos[0]

            for j in range(1, int(number_of_intervals)):
                rm_vec[i * number_of_intervals + j] = (
                    rm_pos[j] * mass_value[j] - rm_pos[j - 1] * mass_value[j - 1]
                ) / (mass_value[j] - mass_value[j - 1])

            mass_vec[i * number_of_intervals] = mass_value[0]

            for j in range(1, int(number_of_intervals)):
                mass_vec[i * number_of_intervals + j] = (
                    mass_value[j] - mass_value[j - 1]
                )

        self.mass_vector = mass_vec
        self.rm_vector = rm_vec

    @staticmethod
    @njit(cache=True, parallel=True)
    def mascon_potential(position, mass_vector, rm_vector, G):
        """
        Calculates the gravitational potential energy due to a set of mascons.

        Parameters:
        position (list): The position vector of the point where the potential is calculated.
        mass_vector (list): The vector containing the masses of the mascons.
        rm_vector (list): The vector containing the positions of the mascons.
        G (float): The gravitational constant.

        Returns:
        float: The gravitational potential energy.

        """
        U = 0
        for i in prange(len(mass_vector)):
            radius = np.sqrt(
                (position[0][0] - rm_vector[i][0]) ** 2
                + (position[0][1] - rm_vector[i][1]) ** 2
                + (position[0][2] - rm_vector[i][2]) ** 2
            )
            U -= G * mass_vector[i] / radius
        return U

    @staticmethod
    @njit(cache=True, parallel=True)
    def mascon_acceleration(position, mass_vector, rm_vector, G):
        """
        Calculates the acceleration due to gravity caused by a collection of mass vectors (mascons) on a given position.

        Parameters:
        position (numpy.ndarray): The position vector at which the acceleration is calculated.
        mass_vector (numpy.ndarray): The array of mass vectors representing the mascons.
        rm_vector (numpy.ndarray): The array of position vectors representing the mascons.
        G (float): The gravitational constant.

        Returns:
        numpy.ndarray: The acceleration vector due to gravity caused by the mascons at the given position.
        """
        acc = np.zeros(3)
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

    def calculate_potential(self, position):
        """
        Calculates the potential at a given position due to the mascon.

        Parameters:
        position (tuple): The position at which to calculate the potential.

        Returns:
        float: The potential at the given position.
        """
        return self.mascon_potential(position, self.mass_vector, self.rm_vector, self.G)

    def calculate_acceleration(self, position):
        """
        Calculates the acceleration at a given position due to the mascon.

        Parameters:
        position (tuple): The position at which to calculate the acceleration.

        Returns:
        tuple: The acceleration vector at the given position.
        """
        return self.mascon_acceleration(
            position, self.mass_vector, self.rm_vector, self.G
        )
