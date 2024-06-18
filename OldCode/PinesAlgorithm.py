import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit, njit, prange


# PINES ALGORITHM:
# POTENTIAL AND ACCELERATION COMPUTATION GIVEN NORMALIZED COEFFICIENTS


# Conversion from Cartesian to spherical coordinates
def cart2sphPines(x, y, z):
    radius = np.sqrt(x**2 + y**2 + z**2)
    s = x / radius
    t = y / radius
    u = z / radius
    return radius, s, t, u


#
def HelmPoly(nmax, u):
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
                        np.sqrt((1 + delta) * (2 * n + 1) / 2 / n) * Hvec[count - n - 1]
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
    return Hvec


def HelmPolyDer(nmax, u, Hvec):
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
    return dHvec


# usalo se devi normalizzare i coefficienti
def Coeff_norm(nmax, C):
    C_norm = C
    if nmax > 0:
        count = 1
        for n in range(1, nmax + 1):
            for m in range(0, n + 1):
                n = int(n)
                m = int(m)
                if m == 0:
                    delta = 1
                else:
                    delta = 0
                Norm = np.sqrt(
                    (2 - delta)
                    * (2 * n + 1)
                    * math.factorial(n - m)
                    / math.factorial(n + m)
                )
                C_norm[count] = C[count] / Norm
                count += 1
    return C_norm


def ParalFact(R, r, mass, nmax):
    rho_vec = np.zeros(nmax + 1)
    G = 6.67408e-11
    rho_vec[0] = G * mass / r
    if nmax > 0:
        for count in range(1, nmax + 1):
            rho_vec[count] = rho_vec[count - 1] * R / r
    return rho_vec


# Pines Algorithm for potential computation
def Pines_pot(R, mass, x, y, z, nmax, C_norm, S_norm):
    # Initialize
    [r, s, t, u] = cart2sphPines(x, y, z)
    th = np.arctan2(z, np.sqrt(x**2 + y**2))
    lam = np.arctan2(y, x)
    rho_vec = ParalFact(R, r, mass, nmax)
    Hvec = HelmPoly(nmax, u)
    U = 0

    # Compute potential
    for m in range(0, nmax + 1):
        for n in range(m, nmax + 1):
            n = int(n)
            m = int(m)
            count = int((n + 1) * (n + 2) / 2) - n + m - 1
            U = U + (
                (
                    rho_vec[n] * C_norm[count] * Hvec[count] * math.cos(m * lam)
                    + rho_vec[n] * S_norm[count] * Hvec[count] * math.sin(m * lam)
                )
                * (math.cos(th)) ** m
            )

    return -U


# Pines Algorithm for acceleration computation
@njit(cache=True, parallel=True)
def Pines_acc(R, mass, x, y, z, nmax, C_norm, S_norm):
    r = np.sqrt(x**2 + y**2 + z**2)
    s = x / r
    t = y / r
    u = z / r
    th = np.arctan2(z, np.sqrt(x**2 + y**2))
    lam = np.arctan2(y, x)

    rho_vec = np.zeros(nmax + 1)
    G = 6.67408e-11
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
                        np.sqrt((1 + delta) * (2 * n + 1) / 2 / n) * Hvec[count - n - 1]
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
                        rho_vec[n] * C_norm[count] * dHvec[count] * math.cos(m * lam)
                        + rho_vec[n] * S_norm[count] * dHvec[count] * math.sin(m * lam)
                    )
                    * (math.cos(th)) ** m
                )
                a_4 = (
                    a_4
                    - (
                        rho_vec[n]
                        * C_norm[count]
                        * ((n + m + 1) * Hvec[count] + u * dHvec[count])
                        * math.cos(m * lam)
                        + rho_vec[n]
                        * S_norm[count]
                        * ((n + m + 1) * Hvec[count] + u * dHvec[count])
                        * math.sin(m * lam)
                    )
                    * (math.cos(th)) ** m
                )
            else:
                a_1 = (
                    a_1
                    + (
                        rho_vec[n]
                        * C_norm[count]
                        * Hvec[count]
                        * math.cos((m - 1) * lam)
                        + rho_vec[n]
                        * S_norm[count]
                        * Hvec[count]
                        * math.sin((m - 1) * lam)
                    )
                    * (math.cos(th)) ** (m - 1)
                    * m
                )
                a_2 = (
                    a_2
                    + (
                        -rho_vec[n]
                        * C_norm[count]
                        * Hvec[count]
                        * math.sin((m - 1) * lam)
                        + rho_vec[n]
                        * S_norm[count]
                        * Hvec[count]
                        * math.cos((m - 1) * lam)
                    )
                    * (math.cos(th)) ** (m - 1)
                    * m
                )
                a_3 = (
                    a_3
                    + (
                        rho_vec[n] * C_norm[count] * dHvec[count] * math.cos(m * lam)
                        + rho_vec[n] * S_norm[count] * dHvec[count] * math.sin(m * lam)
                    )
                    * (math.cos(th)) ** m
                )
                a_4 = (
                    a_4
                    - (
                        rho_vec[n]
                        * C_norm[count]
                        * ((n + m + 1) * Hvec[count] + u * dHvec[count])
                        * math.cos(m * lam)
                        + rho_vec[n]
                        * S_norm[count]
                        * ((n + m + 1) * Hvec[count] + u * dHvec[count])
                        * math.sin(m * lam)
                    )
                    * (math.cos(th)) ** m
                )
        acc = np.array([a_1 + s * a_4, a_2 + t * a_4, a_3 + u * a_4]) / r

    return acc
