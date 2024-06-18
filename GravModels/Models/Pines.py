import numpy as np
import math
from numba import njit
import math
import os
import numpy as np
import trimesh
from GravModels.CelestialBodies.Asteroids import *
from GravModels.utils.ProgressBar import ProgressBar


class Pines:
    def __init__(self, asteroid, nmax=4):
        """
        Initialize a Pines object.

        Parameters:
        - asteroid: An instance of the Asteroid class representing the asteroid.
        - nmax: An integer representing the maximum degree of the spherical harmonics expansion (default is 4).

        Attributes:
        - R: The radius of the asteroid.
        - mass: The mass of the asteroid.
        - nmax: The maximum degree of the spherical harmonics expansion.
        - C_nm: The normalized cosine coefficients of the spherical harmonics expansion.
        - S_nm: The normalized sine coefficients of the spherical harmonics expansion.
        - G: The gravitational constant.

        Returns:
        None
        """
        self.R = asteroid.radius
        self.mass = asteroid.mass
        self.nmax = nmax
        self.C_nm, self.S_nm = SphericalHarmonics(
            asteroid, nmax
        ).normalized_coefficients_calculate()
        self.G = 6.67408e-11

    @staticmethod
    def cart2sph(position):
        """
        Convert Cartesian coordinates to spherical coordinates.

        Args:
            position (numpy.ndarray): Array of shape (1, 3) representing the Cartesian coordinates.

        Returns:
            tuple: A tuple containing the spherical coordinates in the following order:
                - radius (float): The distance from the origin to the point.
                - s (float): The sine of the polar angle.
                - t (float): The tangent of the azimuthal angle.
                - u (float): The cotangent of the polar angle.
                - theta (float): The polar angle in radians.
                - lambd (float): The azimuthal angle in radians.
        """
        x, y, z = position[0][0], position[0][1], position[0][2]
        radius = np.sqrt(x**2 + y**2 + z**2)
        s = x / radius
        t = y / radius
        u = z / radius
        theta = np.arctan2(z, np.sqrt(x**2 + y**2))
        lambd = np.arctan2(y, x)
        return radius, s, t, u, theta, lambd

    @staticmethod
    def helmholtz_poly(nmax, u):
        """
        Calculate the Helmoltz polynomials up to a given maximum degree.

        Parameters:
        - nmax (int): The maximum degree of the Helmoltz polynomials.
        - u (float): The input value for the polynomials.

        Returns:
        - Hvec (numpy.ndarray): An array containing the calculated Helmoltz polynomials.

        The function calculates the Helmoltz polynomials up to the given maximum degree
        using the input value u. The result is returned as an array.
        """
        Hvec = np.zeros(int((nmax + 1) * (nmax + 2) / 2))
        Hvec[0] = 1

        if nmax > 0:
            for n in range(1, nmax + 1):
                count = int((n + 1) * (n + 2) / 2) - 1
                for m in range(n, -1, -1):
                    if n == m:
                        delta = 1 if n == 1 else 0
                        Hvec[count] = (
                            np.sqrt((1 + delta) * (2 * n + 1) / (2 * n))
                            * Hvec[count - n - 1]
                        )
                    elif m == n - 1:
                        delta = 1 if n == 1 else 0
                        Hvec[count - 1] = u * np.sqrt(2 * n / (1 + delta)) * Hvec[count]
                    else:
                        delta = 1 if m == 0 else 0
                        Hvec[count - n + m] = np.sqrt(
                            (n + m + 1) / ((n - m) * (1 + delta))
                        ) * u * Hvec[count - n + m + 1] - Hvec[
                            count - 2 * n + m + 1
                        ] * np.sqrt(
                            ((2 * n + 1) * (n - m - 1))
                            / ((1 + delta) * (2 * n - 1) * (n - m))
                        )
        return Hvec

    @staticmethod
    def helmholtz_poly_der(nmax, u, Hvec):
        """
        Calculate the derivative of the Helmoltz polynomials.

        Parameters:
        - nmax (int): The maximum degree of the polynomials.
        - u (float): The input variable.
        - Hvec (numpy.ndarray): The array of Helmoltz polynomials.

        Returns:
        - dHvec (numpy.ndarray): The array of derivatives of the Helmoltz polynomials.
        """
        dHvec = np.zeros(int((nmax + 1) * (nmax + 2) / 2))
        dHvec[0] = 0

        if nmax > 0:
            count = 1
            for n in range(1, nmax + 1):
                for m in range(n + 1):
                    if n == m:
                        dHvec[count] = 0
                    else:
                        delta = 1 if m == 0 else 0
                        dHvec[count] = (
                            np.sqrt((2 - delta) * (n - m) * (n + m + 1) / 2)
                            * Hvec[count + 1]
                        )
                    count += 1
        return dHvec

    def parallel_factors(self, r):
        """
        Calculates the parallel factors for a given distance 'r'.

        Parameters:
        - r (float): The distance at which to calculate the parallel factors.

        Returns:
        - rho_vec (numpy.ndarray): An array containing the calculated parallel factors.

        """
        rho_vec = np.zeros(self.nmax + 1)
        rho_vec[0] = self.G * self.mass / r
        if self.nmax > 0:
            for count in range(1, self.nmax + 1):
                rho_vec[count] = rho_vec[count - 1] * self.R / r
        return rho_vec

    def calculate_potential(self, position):
        """
        Calculates the potential at a given position.

        Parameters:
        - position: A tuple or list containing the Cartesian coordinates (x, y, z) of the position.

        Returns:
        - The potential at the given position.

        """
        radius, s, t, u, theta, lambd = self.cart2sph(position)
        rho_vec = self.parallel_factors(radius)
        Hvec = self.helmholtz_poly(self.nmax, u)
        U = 0

        for m in range(self.nmax + 1):
            for n in range(m, self.nmax + 1):
                count = int((n + 1) * (n + 2) / 2) - n + m - 1
                U += (
                    rho_vec[n] * self.C_nm[count] * Hvec[count] * np.cos(m * lambd)
                    + rho_vec[n] * self.S_nm[count] * Hvec[count] * np.sin(m * lambd)
                ) * (np.cos(theta)) ** m

        return -U

    @staticmethod
    @njit(cache=True, parallel=True)
    def pines_acceleration(R, mass, position, nmax, C_nm, S_nm, G):
        """
        Calculate the acceleration due to gravity using the Pines model.

        Parameters:
        - R (float): The radius of the celestial body.
        - mass (float): The mass of the celestial body.
        - position (list): The position vector of the object in Cartesian coordinates.
        - nmax (int): The maximum degree of the spherical harmonics expansion.
        - C_nm (list): The cosine coefficients of the spherical harmonics expansion.
        - S_nm (list): The sine coefficients of the spherical harmonics expansion.
        - G (float): The gravitational constant.

        Returns:
        - acc (numpy.ndarray): The acceleration vector in Cartesian coordinates.
        """

        x, y, z = position[0][0], position[0][1], position[0][2]
        r = np.sqrt(x**2 + y**2 + z**2)
        s = x / r
        t = y / r
        u = z / r
        theta = np.arctan2(z, np.sqrt(x**2 + y**2))
        lambd = np.arctan2(y, x)

        rho_vec = np.zeros(nmax + 1)
        rho_vec[0] = G * mass / r
        if nmax > 0:
            for count in range(1, nmax + 1):
                rho_vec[count] = rho_vec[count - 1] * R / r

        Hvec = np.zeros(int((nmax + 1) * (nmax + 2) / 2))

        Hvec[0] = 1

        if nmax > 0:
            for n in range(1, nmax + 1):
                count = int((n + 1) * (n + 2) / 2) - 1
                for m in np.linspace(n, 0, n + 1):
                    n = int(n)
                    m = int(m)
                    if n == m:
                        if n == 1:
                            delta = 1
                        else:
                            delta = 0
                        Hvec[count] = (
                            np.sqrt((1 + delta) * (2 * n + 1) / 2 / n)
                            * Hvec[count - n - 1]
                        )
                    elif m == n - 1:
                        if n == 1:
                            delta = 1
                        else:
                            delta = 0
                        Hvec[count - 1] = u * np.sqrt(2 * n / (1 + delta)) * Hvec[count]
                    else:
                        if m == 0:
                            delta = 1
                        else:
                            delta = 0
                        Hvec[count - n + m] = np.sqrt(
                            (n + m + 1) / ((n - m) * (1 + delta))
                        ) * u * Hvec[count - n + m + 1] - Hvec[
                            count - 2 * n + m + 1
                        ] * np.sqrt(
                            ((2 * n + 1) * (n - m - 1))
                            / ((1 + delta) * (2 * n - 1) * (n - m))
                        )

        dHvec = np.zeros(int((nmax + 1) * (nmax + 2) / 2))

        dHvec[0] = 0

        if nmax > 0:
            count = 1
            for n in range(1, nmax + 1):
                for m in range(0, n + 1):
                    n = int(n)
                    m = int(m)
                    if n == m:
                        dHvec[count] = 0
                    else:
                        if m == 0:
                            delta = 1
                        else:
                            delta = 0
                        dHvec[count] = (
                            np.sqrt((2 - delta) * (n - m) * (n + m + 1) / 2)
                            * Hvec[count + 1]
                        )
                    count += 1

        a_1 = 0
        a_2 = 0
        a_3 = 0
        a_4 = 0

        for m in range(0, nmax + 1):
            for n in range(m, nmax + 1):
                n = int(n)
                m = int(m)
                count = int((n + 1) * (n + 2) / 2) - n + m - 1
                if m == 0:
                    a_3 = (
                        a_3
                        + (
                            rho_vec[n]
                            * C_nm[count]
                            * dHvec[count]
                            * math.cos(m * lambd)
                            + rho_vec[n]
                            * S_nm[count]
                            * dHvec[count]
                            * math.sin(m * lambd)
                        )
                        * (math.cos(theta)) ** m
                    )
                    a_4 = (
                        a_4
                        - (
                            rho_vec[n]
                            * C_nm[count]
                            * ((n + m + 1) * Hvec[count] + u * dHvec[count])
                            * math.cos(m * lambd)
                            + rho_vec[n]
                            * S_nm[count]
                            * ((n + m + 1) * Hvec[count] + u * dHvec[count])
                            * math.sin(m * lambd)
                        )
                        * (math.cos(theta)) ** m
                    )
                else:
                    a_1 = (
                        a_1
                        + (
                            rho_vec[n]
                            * C_nm[count]
                            * Hvec[count]
                            * math.cos((m - 1) * lambd)
                            + rho_vec[n]
                            * S_nm[count]
                            * Hvec[count]
                            * math.sin((m - 1) * lambd)
                        )
                        * (math.cos(theta)) ** (m - 1)
                        * m
                    )
                    a_2 = (
                        a_2
                        + (
                            -rho_vec[n]
                            * C_nm[count]
                            * Hvec[count]
                            * math.sin((m - 1) * lambd)
                            + rho_vec[n]
                            * S_nm[count]
                            * Hvec[count]
                            * math.cos((m - 1) * lambd)
                        )
                        * (math.cos(theta)) ** (m - 1)
                        * m
                    )
                    a_3 = (
                        a_3
                        + (
                            rho_vec[n]
                            * C_nm[count]
                            * dHvec[count]
                            * math.cos(m * lambd)
                            + rho_vec[n]
                            * S_nm[count]
                            * dHvec[count]
                            * math.sin(m * lambd)
                        )
                        * (math.cos(theta)) ** m
                    )
                    a_4 = (
                        a_4
                        - (
                            rho_vec[n]
                            * C_nm[count]
                            * ((n + m + 1) * Hvec[count] + u * dHvec[count])
                            * math.cos(m * lambd)
                            + rho_vec[n]
                            * S_nm[count]
                            * ((n + m + 1) * Hvec[count] + u * dHvec[count])
                            * math.sin(m * lambd)
                        )
                        * (math.cos(theta)) ** m
                    )
            acc = np.array([a_1 + s * a_4, a_2 + t * a_4, a_3 + u * a_4]) / r

        return acc

    def calculate_acceleration(self, position):
        """
        Calculates the acceleration at a given position using the Pines model.

        Parameters:
        - position: The position at which to calculate the acceleration.

        Returns:
        - The acceleration at the given position.
        """
        return self.pines_acceleration(
            self.R, self.mass, position, self.nmax, self.C_nm, self.S_nm, self.G
        )


class SphericalHarmonics:
    def __init__(self, asteroid, nmax):
        """
        Initialize a Pines object.

        Parameters:
        - asteroid: An instance of the Asteroid class.
        - nmax: The maximum value for n.

        Attributes:
        - nmax: The maximum value for n.
        - asteroid: An instance of the Asteroid class.
        - density: The density of the asteroid.
        - mass: The mass of the asteroid.
        - radius: The radius of the asteroid.
        - model: The model of the asteroid.
        - G: The gravitational constant.
        - scaleFactor: The scale factor.
        - mesh: The loaded mesh.

        Returns:
        None
        """

        self.nmax = nmax
        self.asteroid = asteroid
        self.density = asteroid.density
        self.mass = asteroid.mass
        self.radius = asteroid.radius
        self.model = asteroid.model
        self.G = 6.67408e-11
        self.scaleFactor = 1e3
        self.mesh = self.load_mesh()

    def load_mesh(self):
        """
        Loads a mesh from the specified file.

        Returns:
            The loaded mesh.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file extension is not supported.
        """
        _, file_extension = os.path.splitext(self.model)
        return trimesh.load_mesh(self.model, file_type=file_extension[1:])

    @staticmethod
    def cart2sph(position):
        """
        Convert Cartesian coordinates to spherical coordinates.

        Parameters:
        position (numpy.ndarray): Array of shape (1, 3) representing the Cartesian coordinates (x, y, z).

        Returns:
        tuple: A tuple containing the spherical coordinates (radius, theta, lambd).
            - radius (float): The radial distance from the origin to the point.
            - theta (float): The polar angle in radians, measured from the positive z-axis.
            - lambd (float): The azimuthal angle in radians, measured from the positive x-axis.
        """
        x, y, z = position[0][0], position[0][1], position[0][2]
        xy = x**2 + y**2
        radius = np.sqrt(xy + z**2)
        theta = np.arctan2(z, np.sqrt(xy))
        lambd = np.arctan2(y, x)
        return radius, theta, lambd

    @staticmethod
    def legendre_poly(nmax, theta):
        """
        Calculate the Legendre polynomials up to a given order.

        Parameters:
        - nmax (int): The maximum order of the Legendre polynomials.
        - theta (float): The angle at which to evaluate the Legendre polynomials.

        Returns:
        - Pvec (numpy.ndarray): An array containing the Legendre polynomials up to order nmax.

        """
        Pvec = np.zeros(int((nmax + 1) * (nmax + 2) / 2))

        Pvec[0] = 1
        if nmax > 0:
            Pvec[1] = math.sin(theta)
            Pvec[2] = math.cos(theta)

        count = 3
        if nmax > 1:
            for i in range(2, nmax + 1):
                for j in range(i + 1):
                    if j == i - 1:
                        Pvec[count + 1] = (
                            (2 * i - 1) * math.cos(theta) * Pvec[count - i]
                        )
                        Pvec[count] = (
                            math.sin(theta) / (1 - math.sin(theta) ** 2) ** 0.5
                        ) * Pvec[count + 1]
                    elif j != i:
                        Pvec[count] = (
                            (2 * i - 1) * math.sin(theta) * Pvec[count - i]
                            - (i + j - 1) * Pvec[count - 2 * i + 1]
                        ) / (i - j)
                    count += 1
        return Pvec

    def calculate_potential(self, x, y, z, C, S):
        """
        Calculates the gravitational potential at a given point (x, y, z) due to the gravitational field of the object.

        Parameters:
        - x (float): The x-coordinate of the point.
        - y (float): The y-coordinate of the point.
        - z (float): The z-coordinate of the point.
        - C (list): List of coefficients for the cosine terms in the potential expansion.
        - S (list): List of coefficients for the sine terms in the potential expansion.

        Returns:
        - U (float): The gravitational potential at the given point.

        """
        r, theta, lambd = self.cart2sph(x, y, z)
        Pvec = self.legendre_poly(self.nmax, theta)
        U = 1

        if self.nmax > 0:
            count = 1
            for n in range(1, self.nmax + 1):
                for m in range(0, n + 1):
                    U += (self.radius / r) ** n * (
                        Pvec[count]
                        * (
                            C[count] * math.cos(lambd * m)
                            + S[count] * math.sin(lambd * m)
                        )
                    )
                    count += 1

        U = -U * self.G * self.mass / r
        return U

    @staticmethod
    def calculate_product(input_string):
        """
        Calculates the product of the occurrences of each character in the input string.

        Args:
            input_string (str): The input string to calculate the product for.

        Returns:
            int: The product of the occurrences of each character in the input string.
        """
        char_count = {}

        # Count the occurrences of each character
        for char in input_string:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1

        # Initialize the product to 1
        product = 1

        # Calculate the product of repeating characters
        for count in char_count.values():
            product *= count

        return product

    def normalized_coefficients_calculate(self):
        """
        Calculates the normalized coefficients for the Pines model.

        Returns:
            None
        """
        C_nm = np.zeros(int((self.nmax + 1) * (self.nmax + 2) / 2))
        S_nm = np.zeros(int((self.nmax + 1) * (self.nmax + 2) / 2))
        sum_c = np.zeros(int((self.nmax + 1) * (self.nmax + 2) / 2))
        sum_s = np.zeros(int((self.nmax + 1) * (self.nmax + 2) / 2))

        bar = ProgressBar(len(self.mesh.faces))
        for i in range(len(self.mesh.faces)):
            r1 = self.mesh.vertices[int(self.mesh.faces[i][0])] * self.scaleFactor
            r2 = self.mesh.vertices[int(self.mesh.faces[i][1])] * self.scaleFactor
            r3 = self.mesh.vertices[int(self.mesh.faces[i][2])] * self.scaleFactor

            x_dot = [r1[0], r2[0], r3[0]]
            y_dot = [r1[1], r2[1], r3[1]]
            z_dot = [r1[2], r2[2], r3[2]]

            J = np.array(
                [[r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]]]
            )
            det = np.linalg.det(J)

            sum_c[0] += det / math.factorial(3)

            if self.nmax > 0:
                sum_c[2] += (
                    det
                    / math.factorial(4)
                    * (r1[0] + r2[0] + r3[0])
                    / self.radius
                    / math.sqrt(3)
                )
                sum_c[1] += (
                    det
                    / math.factorial(4)
                    * (r1[2] + r2[2] + r3[2])
                    / self.radius
                    / math.sqrt(3)
                )
                sum_s[2] += (
                    det
                    / math.factorial(4)
                    * (r1[1] + r2[1] + r3[1])
                    / self.radius
                    / math.sqrt(3)
                )

                vec_c_nn_old = [
                    r1[0] / self.radius / math.sqrt(3),
                    r2[0] / self.radius / math.sqrt(3),
                    r3[0] / self.radius / math.sqrt(3),
                ]
                vec_s_nn_old = [
                    r1[1] / self.radius / math.sqrt(3),
                    r2[1] / self.radius / math.sqrt(3),
                    r3[1] / self.radius / math.sqrt(3),
                ]

                vec_c_nm_old_1 = [
                    [
                        r1[2] / math.sqrt(3) / self.radius,
                        r2[2] / math.sqrt(3) / self.radius,
                        r3[2] / math.sqrt(3) / self.radius,
                    ],
                    [
                        r1[0] / math.sqrt(3) / self.radius,
                        r2[0] / math.sqrt(3) / self.radius,
                        r3[0] / math.sqrt(3) / self.radius,
                    ],
                ]

                vec_s_nm_old_1 = [
                    [0, 0, 0],
                    [
                        r1[1] / math.sqrt(3) / self.radius,
                        r2[1] / math.sqrt(3) / self.radius,
                        r3[1] / math.sqrt(3) / self.radius,
                    ],
                ]

                vec_c_nm_old_2 = [[1]]
                vec_s_nm_old_2 = [[0]]

                vec_nn_rep_old = ["1", "2", "3"]
                vec_nn_rep_old_1 = ["1", "2", "3"]
                vec_nn_rep_old_2 = [""]

            if self.nmax > 1:
                count = 3
                for n in range(2, self.nmax + 1):
                    vec_c_nm_new_1_tot = []
                    vec_s_nm_new_1_tot = []
                    for m in range(0, n + 1):
                        if m < n - 1:
                            vec_c_nm_new_1 = np.zeros(3**n)
                            vec_s_nm_new_1 = np.zeros(3**n)
                            vec_c_nm_new_2 = np.zeros(3**n)
                            vec_s_nm_new_2 = np.zeros(3**n)
                            vec_c_n_1m = vec_c_nm_old_1[m]
                            vec_s_n_1m = vec_s_nm_old_1[m]
                            vec_c_n_2m = vec_c_nm_old_2[m]
                            vec_s_n_2m = vec_s_nm_old_2[m]
                            if m == 0:
                                vec_nn_rep_new_1 = []
                                vec_nn_rep_new_2 = []

                            for j in range(3):
                                for h in range(3):
                                    for k in range(len(vec_c_n_2m)):
                                        const = -math.sqrt(
                                            (2 * n - 3)
                                            * (n + m - 1)
                                            * (n - m - 1)
                                            / ((2 * n + 1) * (n + m) * (n - m))
                                        )
                                        vec_c_nm_new_2[
                                            j * 3 ** (n - 1) + h * 3 ** (n - 2) + k
                                        ] = (
                                            (
                                                x_dot[j] * x_dot[h]
                                                + y_dot[j] * y_dot[h]
                                                + z_dot[j] * z_dot[h]
                                            )
                                            * vec_c_n_2m[k]
                                            / self.radius**2
                                            * const
                                        )
                                        vec_s_nm_new_2[
                                            j * 3 ** (n - 1) + h * 3 ** (n - 2) + k
                                        ] = (
                                            (
                                                x_dot[j] * x_dot[h]
                                                + y_dot[j] * y_dot[h]
                                                + z_dot[j] * z_dot[h]
                                            )
                                            * vec_s_n_2m[k]
                                            / self.radius**2
                                            * const
                                        )

                                        if m == 0:
                                            vec_nn_rep_new_2.append(
                                                vec_nn_rep_old_2[k]
                                                + str(h + 1)
                                                + str(j + 1)
                                            )

                            if m == 0:
                                repet_vector_2 = []
                                for string in vec_nn_rep_new_2:
                                    repet_vector_2.append(
                                        self.calculate_product(string)
                                    )

                            for j in range(3):
                                for k in range(len(vec_c_n_1m)):
                                    const = (2 * n - 1) * math.sqrt(
                                        (2 * n - 1) / ((2 * n + 1) * (n + m) * (n - m))
                                    )
                                    vec_c_nm_new_1[j * 3 ** (n - 1) + k] = (
                                        z_dot[j] * vec_c_n_1m[k] / self.radius * const
                                    )
                                    vec_s_nm_new_1[j * 3 ** (n - 1) + k] = (
                                        z_dot[j] * vec_s_n_1m[k] / self.radius * const
                                    )
                                    if m == 0:
                                        vec_nn_rep_new_1.append(
                                            vec_nn_rep_old_1[k] + str(j + 1)
                                        )

                            if m == 0:
                                vec_nn_rep_old_2 = vec_nn_rep_old_1
                                vec_nn_rep_old_1 = vec_nn_rep_new_1

                                repet_vector_1 = []
                                for string in vec_nn_rep_new_1:
                                    repet_vector_1.append(
                                        self.calculate_product(string)
                                    )

                            vec_c_nm_new_1_tot.append(vec_c_nm_new_1 + vec_c_nm_new_2)
                            vec_s_nm_new_1_tot.append(vec_s_nm_new_1 + vec_s_nm_new_2)

                            summation_c = np.sum(
                                np.array(repet_vector_1) * vec_c_nm_new_1
                            ) + np.sum(np.array(repet_vector_2) * vec_c_nm_new_2)
                            summation_s = np.sum(
                                np.array(repet_vector_1) * vec_s_nm_new_1
                            ) + np.sum(np.array(repet_vector_2) * vec_s_nm_new_2)

                            sum_c[count] += det / math.factorial(n + 3) * summation_c
                            sum_s[count] += det / math.factorial(n + 3) * summation_s

                        if m == n - 1:
                            vec_nn_rep_new = []
                            vec_c_nm_new = np.zeros(3**n)
                            vec_s_nm_new = np.zeros(3**n)
                            for j in range(3):
                                for k in range(len(vec_c_nn_old)):
                                    const = (2 * n - 1) / math.sqrt(2 * n + 1)
                                    vec_c_nm_new[j * 3 ** (n - 1) + k] = (
                                        z_dot[j] * vec_c_nn_old[k] / self.radius * const
                                    )
                                    vec_s_nm_new[j * 3 ** (n - 1) + k] = (
                                        z_dot[j] * vec_s_nn_old[k] / self.radius * const
                                    )
                                    vec_nn_rep_new.append(
                                        str(j + 1) + vec_nn_rep_old[k]
                                    )

                            vec_c_nm_new_1_tot.append(vec_c_nm_new)
                            vec_s_nm_new_1_tot.append(vec_s_nm_new)

                            vec_nn_rep_old = vec_nn_rep_new

                            repet_vector = []
                            for string in vec_nn_rep_new:
                                repet_vector.append(self.calculate_product(string))

                            summation_c = np.sum(np.array(repet_vector) * vec_c_nm_new)
                            summation_s = np.sum(np.array(repet_vector) * vec_s_nm_new)

                            sum_c[count] += det / math.factorial(n + 3) * summation_c
                            sum_s[count] += det / math.factorial(n + 3) * summation_s

                        elif n == m:
                            vec_c_nn_new = np.zeros(3**n)
                            vec_s_nn_new = np.zeros(3**n)
                            for j in range(3):
                                for k in range(len(vec_c_nn_old)):
                                    const = (2 * n - 1) / math.sqrt(2 * n * (2 * n + 1))
                                    vec_c_nn_new[j * 3 ** (n - 1) + k] = (
                                        x_dot[j] * vec_c_nn_old[k] / self.radius
                                        - y_dot[j] * vec_s_nn_old[k] / self.radius
                                    ) * const
                                    vec_s_nn_new[j * 3 ** (n - 1) + k] = (
                                        y_dot[j] * vec_c_nn_old[k] / self.radius
                                        + x_dot[j] * vec_s_nn_old[k] / self.radius
                                    ) * const
                            vec_c_nn_old = vec_c_nn_new
                            vec_s_nn_old = vec_s_nn_new

                            vec_c_nm_new_1_tot.append(vec_c_nn_new)
                            vec_s_nm_new_1_tot.append(vec_s_nn_new)

                            summation_c = np.sum(np.array(repet_vector) * vec_c_nn_new)
                            summation_s = np.sum(np.array(repet_vector) * vec_s_nn_new)

                            sum_c[count] += det / math.factorial(n + 3) * summation_c
                            sum_s[count] += det / math.factorial(n + 3) * summation_s

                            vec_c_nm_old_2 = vec_c_nm_old_1
                            vec_s_nm_old_2 = vec_c_nm_old_2
                            vec_c_nm_old_1 = vec_c_nm_new_1_tot
                            vec_s_nm_old_1 = vec_s_nm_new_1_tot

                        count += 1
            bar.update(i)
        bar.markComplete()
        bar.close()

        C_nm[0] = self.density / self.mass

        if self.nmax > 0:
            C_nm[2] = self.density / self.mass
            C_nm[1] = self.density / self.mass
            S_nm[2] = self.density / self.mass

        if self.nmax > 1:
            count = 3
            for n in range(2, self.nmax + 1):
                for m in range(0, n + 1):
                    C_nm[count] = self.density / self.mass
                    if m == 0:
                        S_nm[count] = 0
                    else:
                        S_nm[count] = self.density / self.mass
                    count += 1

        C_nm *= sum_c
        S_nm *= sum_s
        return C_nm, S_nm
