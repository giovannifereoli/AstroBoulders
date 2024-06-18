import numpy as np
import math


# Codice per calcolare lo SH potential, Pines è più comodo per calcolare l'accelerazione (è sempre SH)
def cart2sph(x, y, z):
    xy = x**2 + y**2
    radius = np.sqrt(xy + z**2)
    theta = np.arctan2(z, np.sqrt(xy))
    lambd = np.arctan2(y, x)
    return radius, theta, lambd


def LegenPoly(nmax, theta):
    Pvec = np.zeros(int((nmax + 1) * (nmax + 2) / 2))

    Pvec[0] = 1
    if nmax > 0:
        Pvec[1] = math.sin(theta)
        Pvec[2] = math.cos(theta)

    count = 3
    if nmax > 1:
        for i in range(2, nmax + 1):
            for j in range(i + 1):
                i = int(i)
                j = int(j)
                if j == i - 1:
                    Pvec[count + 1] = (2 * i - 1) * math.cos(theta) * Pvec[count - i]
                    # https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/gravity-SphericalHarmonics.pdf
                    Pvec[count] = (
                        math.sin(theta) / (1 - math.sin(theta) ** 2) ** 0.5
                    ) * Pvec[count + 1]
                elif j != i:
                    Pvec[count] = (
                        (2 * i - 1) * math.sin(theta) * Pvec[count - i]
                        - (i + j - 1) * Pvec[count - 2 * i + 1]
                    ) / (i - j)
                count = count + 1
    return Pvec


def SphHarm_pot(R, mass, x, y, z, nmax, C, S):
    G = 6.67408e-11
    [r, th, lam] = cart2sph(x, y, z)
    Pvec = LegenPoly(nmax, th)
    U = 1

    if nmax > 0:
        count = 1
        for n in range(1, nmax + 1):
            for m in range(0, n + 1):
                U = U + (R / r) ** n * (
                    Pvec[count]
                    * (C[count] * math.cos(lam * m) + S[count] * math.sin(lam * m))
                )
                count = count + 1

    U = -U * G * mass / r
    return U
