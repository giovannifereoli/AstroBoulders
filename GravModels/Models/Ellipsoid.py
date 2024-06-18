import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import math


class Ellipsoid:
    def __init__(self, asteroid):
        """
        Initializes an Ellipsoid object.

        Parameters:
        asteroid (Asteroid): The asteroid object containing the necessary parameters.

        Attributes:
        mu (float): The gravitational parameter of the asteroid.
        a (float): The semi-major axis of the ellipsoid.
        b (float): The semi-minor axis of the ellipsoid.
        c (float): The semi-minor axis of the ellipsoid.
        density (float): The density of the asteroid.
        """

        self._validate_asteroid(asteroid)
        self.mu = asteroid.mu
        self.a = asteroid.a
        self.b = asteroid.b
        self.c = asteroid.c
        self.density = asteroid.density

    def _validate_asteroid(self, asteroid):
        """
        Validates the asteroid object to ensure it has all the required parameters.

        Args:
            asteroid (Asteroid): The asteroid object to be validated.

        Raises:
            ValueError: If any of the required parameters are missing in the asteroid object.
        """
        required_params = ["mu", "a", "b", "c", "density"]
        for param in required_params:
            if not hasattr(asteroid, param):
                raise ValueError(
                    f"Missing required parameter {param} in Asteroid object."
                )

    def potential(self, point):
        """
        Calculates the gravitational potential at a given point due to the ellipsoid.

        Parameters:
        - point (tuple): The coordinates (x, y, z) of the point.

        Returns:
        - potential (float): The gravitational potential at the given point.

        """
        x, y, z = point

        if x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2 - 1 > 0:
            r = np.linalg.norm([x, y, z])
            n = (r / self.a) ** 0.25 if r > 35000 else (r / self.a) ** 0.24
            if r > 50000:
                n = (r / self.a) ** 0.20

            def ext_ellips(k):
                e = (
                    x**2 / (k + self.a**2)
                    + y**2 / (k + self.b**2)
                    + z**2 / (k + self.c**2)
                    - 1
                )
                return e

            root = fsolve(ext_ellips, 0.0, maxfev=10000, xtol=1e-12)
            root = root / (self.a**n)
            if root < 0:
                print("The root of ellipsoid is negative")
                return 0.0

            def integr(s):
                e = (
                    1
                    - x**2 / self.a**n / (self.a**2 / self.a**n + s)
                    - y**2 / self.a**n / (self.b**2 / self.a**n + s)
                    - z**2 / self.a**n / (self.c**2 / self.a**n + s)
                ) / (
                    np.sqrt(
                        (self.a**2 / self.a**n + s)
                        * (self.b**2 / self.a**n + s)
                        * (self.c**2 / self.a**n + s)
                    )
                )
                return e

            potential, _ = quad(integr, root, np.inf, epsabs=1.49e-10, epsrel=1.49e-10)

            return -potential * self.mu * 3 / 4 * self.a ** (-n / 2)
        else:

            def integr(s):
                e = (
                    1
                    - x**2 / (self.a**2 + s)
                    - y**2 / (self.b**2 + s)
                    - z**2 / (self.c**2 + s)
                ) / (np.sqrt((self.a**2 + s) * (self.b**2 + s) * (self.c**2 + s)))
                return e

            potential, _ = quad(integr, 0, np.inf, epsabs=1.49e-10, epsrel=1.49e-10)

            return -potential * self.mu * 3 / 4

    def acceleration(self, point):
        """
        Calculates the acceleration at a given point due to the gravitational field of the ellipsoid.

        Parameters:
            point (tuple): The coordinates (x, y, z) of the point at which the acceleration is to be calculated.

        Returns:
            list: The acceleration vector [acc_x, acc_y, acc_z] at the given point.

        Raises:
            None

        """
        x, y, z = point

        if x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2 - 1 > 0:
            r = np.linalg.norm([x, y, z])
            n = (r / self.a) ** 0.25 if r > 35000 else (r / self.a) ** 0.24
            if r > 50000:
                n = (r / self.a) ** 0.20

            def ext_ellips(k):
                e = (
                    x**2 / (k + self.a**2)
                    + y**2 / (k + self.b**2)
                    + z**2 / (k + self.c**2)
                    - 1
                )
                return e

            root = fsolve(ext_ellips, 0.0, maxfev=10000, xtol=1e-12)
            root = root / (self.a**n)
            if root < 0:
                print("The root of ellipsoid is negative")
                return [0.0, 0.0, 0.0]

            def integr_x(s):
                e = (
                    1
                    / (
                        np.sqrt(
                            (self.a**2 / self.a**n + s)
                            * (self.b**2 / self.a**n + s)
                            * (self.c**2 / self.a**n + s)
                        )
                    )
                    / (self.a**2 / self.a**n + s)
                )
                return e

            def integr_y(s):
                e = (
                    1
                    / (
                        np.sqrt(
                            (self.a**2 / self.a**n + s)
                            * (self.b**2 / self.a**n + s)
                            * (self.c**2 / self.a**n + s)
                        )
                    )
                    / (self.b**2 / self.a**n + s)
                )
                return e

            def integr_z(s):
                e = (
                    1
                    / (
                        np.sqrt(
                            (self.a**2 / self.a**n + s)
                            * (self.b**2 / self.a**n + s)
                            * (self.c**2 / self.a**n + s)
                        )
                    )
                    / (self.c**2 / self.a**n + s)
                )
                return e

            acc_x, _ = quad(integr_x, root, np.inf, epsabs=1.49e-10, epsrel=1.49e-10)
            acc_y, _ = quad(integr_y, root, np.inf, epsabs=1.49e-10, epsrel=1.49e-10)
            acc_z, _ = quad(integr_z, root, np.inf, epsabs=1.49e-10, epsrel=1.49e-10)
            constant = -3 / 2 * self.mu * self.a ** (-1.5 * n)

            return [acc_x * x * constant, acc_y * y * constant, acc_z * z * constant]

        else:

            def integr_x(s):
                e = (
                    1
                    / (np.sqrt((self.a**2 + s) * (self.b**2 + s) * (self.c**2 + s)))
                    / (self.a**2 + s)
                )
                return e

            def integr_y(s):
                e = (
                    1
                    / (np.sqrt((self.a**2 + s) * (self.b**2 + s) * (self.c**2 + s)))
                    / (self.b**2 + s)
                )
                return e

            def integr_z(s):
                e = (
                    1
                    / (np.sqrt((self.a**2 + s) * (self.b**2 + s) * (self.c**2 + s)))
                    / (self.c**2 + s)
                )
                return e

            acc_x, _ = quad(integr_x, 0, np.inf, epsabs=1.49e-10, epsrel=1.49e-10)
            acc_y, _ = quad(integr_y, 0, np.inf, epsabs=1.49e-10, epsrel=1.49e-10)
            acc_z, _ = quad(integr_z, 0, np.inf, epsabs=1.49e-10, epsrel=1.49e-10)

            constant = -3 / 2 * self.mu

            return [acc_x * x * constant, acc_y * y * constant, acc_z * z * constant]

    def laplacian(self, point):
        """
        Calculates the Laplacian of the gravitational potential at a given point.

        Parameters:
        - point (tuple): A tuple containing the coordinates (x, y, z) of the point.

        Returns:
        - float: The Laplacian of the gravitational potential at the given point.
        """
        x, y, z = point

        if x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2 - 1 > 0:
            return 0.0
        else:
            return -4 * math.pi * self.density * self.mu
