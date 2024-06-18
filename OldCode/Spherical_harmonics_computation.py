import math
import os
import numpy as np
import trimesh
from GravModels.CelestialBodies.Asteroids import *
from GravModels.utils.ProgressBar import ProgressBar
from numba import jit, njit, prange


# codice per calcolare i coefficenti del SH già normalizzati usando lo shape del polyhedral
def calculate_product(input_string):
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


def spher_harm_comput(nmax, asteroid):
    density = asteroid.density
    mass = asteroid.mass
    a = asteroid.radius
    obj_file = asteroid.model
    filename, file_extension = os.path.splitext(obj_file)
    mesh = trimesh.load_mesh(obj_file, file_type=file_extension[1:])
    scaleFactor = 1e3
    C_nm = np.zeros(int((nmax + 1) * (nmax + 2) / 2))
    S_nm = np.zeros(int((nmax + 1) * (nmax + 2) / 2))
    sum_c = np.zeros(int((nmax + 1) * (nmax + 2) / 2))
    sum_s = np.zeros(int((nmax + 1) * (nmax + 2) / 2))

    bar = ProgressBar(len(mesh.faces))
    for i in range(len(mesh.faces)):
        r1 = mesh.vertices[int(mesh.faces[i][0])] * scaleFactor
        r2 = mesh.vertices[int(mesh.faces[i][1])] * scaleFactor
        r3 = mesh.vertices[int(mesh.faces[i][2])] * scaleFactor

        x_dot = [r1[0], r2[0], r3[0]]
        y_dot = [r1[1], r2[1], r3[1]]
        z_dot = [r1[2], r2[2], r3[2]]

        J = np.array([[r1[0], r2[0], r3[0]],
                      [r1[1], r2[1], r3[1]],
                      [r1[2], r2[2], r3[2]]])
        det = np.linalg.det(J)

        sum_c[0] += det / math.factorial(3)

        if nmax > 0:
            sum_c[2] += det / math.factorial(4) * (r1[0] + r2[0] + r3[0]) / a / math.sqrt(3)
            sum_c[1] += det / math.factorial(4) * (r1[2] + r2[2] + r3[2]) / a / math.sqrt(3)
            sum_s[2] += det / math.factorial(4) * (r1[1] + r2[1] + r3[1]) / a / math.sqrt(3)

            vec_c_nn_old = [r1[0] / a / math.sqrt(3), r2[0] / a / math.sqrt(3), r3[0] / a / math.sqrt(3)]
            vec_s_nn_old = [r1[1] / a / math.sqrt(3), r2[1] / a / math.sqrt(3), r3[1] / a / math.sqrt(3)]

            vec_c_nm_old_1 = [[r1[2] / math.sqrt(3) / a, r2[2] / math.sqrt(3) / a, r3[2] / math.sqrt(3) / a],
                              [r1[0] / math.sqrt(3) / a, r2[0] / math.sqrt(3) / a, r3[0] / math.sqrt(3) / a]]

            vec_s_nm_old_1 = [[0, 0, 0],
                              [r1[1] / math.sqrt(3) / a, r2[1] / math.sqrt(3) / a, r3[1] / math.sqrt(3) / a]]

            vec_c_nm_old_2 = [[1]]
            vec_s_nm_old_2 = [[0]]

            vec_nn_rep_old = ['1', '2', '3']
            vec_nn_rep_old_1 = ['1', '2', '3']
            vec_nn_rep_old_2 = ['']

        if nmax > 1:
            count = 3
            for n in range(2, nmax+1):
                vec_c_nm_new_1_tot = []
                vec_s_nm_new_1_tot = []
                for m in range(0, n+1):
                    if m < n-1:
                        vec_c_nm_new_1 = np.zeros(3 ** n)
                        vec_s_nm_new_1 = np.zeros(3 ** n)
                        vec_c_nm_new_2 = np.zeros(3 ** n)
                        vec_s_nm_new_2 = np.zeros(3 ** n)
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
                                    const = -math.sqrt((2*n-3)*(n+m-1)*(n-m-1)/((2*n+1)*(n+m)*(n-m)))
                                    vec_c_nm_new_2[j * 3 ** (n - 1) + h * 3 ** (n - 2) + k] = (x_dot[j] * x_dot[h] + y_dot[j] * y_dot[h] + z_dot[j] * z_dot[h]) * vec_c_n_2m[k] / a ** 2 * const
                                    vec_s_nm_new_2[j * 3 ** (n - 1) + h * 3 ** (n - 2) + k] = (x_dot[j] * x_dot[h] + y_dot[j] * y_dot[h] + z_dot[j] * z_dot[h]) * vec_s_n_2m[k] / a ** 2 * const

                                    if m == 0:
                                        vec_nn_rep_new_2.append(vec_nn_rep_old_2[k] + str(h + 1) + str(j + 1))

                        if m == 0:
                            repet_vector_2 = []
                            for string in vec_nn_rep_new_2:
                                repet_vector_2.append(calculate_product(string))

                        for j in range(3):
                            for k in range(len(vec_c_n_1m)):
                                const = (2*n-1)*math.sqrt((2*n-1)/((2*n+1)*(n+m)*(n-m)))
                                vec_c_nm_new_1[j * 3 ** (n - 1) + k] = z_dot[j] * vec_c_n_1m[k] / a * const
                                vec_s_nm_new_1[j * 3 ** (n - 1) + k] = z_dot[j] * vec_s_n_1m[k] / a * const
                                if m == 0:
                                    vec_nn_rep_new_1.append(vec_nn_rep_old_1[k] + str(j + 1))

                        if m == 0:
                            vec_nn_rep_old_2 = vec_nn_rep_old_1
                            vec_nn_rep_old_1 = vec_nn_rep_new_1

                            repet_vector_1 = []
                            for string in vec_nn_rep_new_1:
                                repet_vector_1.append(calculate_product(string))

                        vec_c_nm_new_1_tot.append(vec_c_nm_new_1 + vec_c_nm_new_2)
                        vec_s_nm_new_1_tot.append(vec_s_nm_new_1 + vec_s_nm_new_2)

                        summation_c = np.sum(np.array(repet_vector_1) * vec_c_nm_new_1)+np.sum(np.array(repet_vector_2) * vec_c_nm_new_2)
                        summation_s = np.sum(np.array(repet_vector_1) * vec_s_nm_new_1)+np.sum(np.array(repet_vector_2) * vec_s_nm_new_2)

                        sum_c[count] += det / math.factorial(n + 3) * summation_c
                        sum_s[count] += det / math.factorial(n + 3) * summation_s

                    if m == n-1:
                        vec_nn_rep_new = []
                        vec_c_nm_new = np.zeros(3**n)
                        vec_s_nm_new = np.zeros(3 ** n)
                        for j in range(3):
                            for k in range(len(vec_c_nn_old)):
                                const = (2*n-1)/math.sqrt(2*n+1)
                                vec_c_nm_new[j*3**(n-1)+k] = z_dot[j]*vec_c_nn_old[k]/a*const
                                vec_s_nm_new[j*3**(n-1)+k] = z_dot[j]*vec_s_nn_old[k]/a*const
                                vec_nn_rep_new.append(str(j + 1) + vec_nn_rep_old[k])

                        vec_c_nm_new_1_tot.append(vec_c_nm_new)
                        vec_s_nm_new_1_tot.append(vec_s_nm_new)

                        vec_nn_rep_old = vec_nn_rep_new

                        repet_vector = []
                        for string in vec_nn_rep_new:
                            repet_vector.append(calculate_product(string))

                        summation_c = np.sum(np.array(repet_vector) * vec_c_nm_new)
                        summation_s = np.sum(np.array(repet_vector) * vec_s_nm_new)

                        sum_c[count] += det / math.factorial(n + 3) * summation_c
                        sum_s[count] += det / math.factorial(n + 3) * summation_s

                    elif n == m:
                        vec_c_nn_new = np.zeros(3 ** n)
                        vec_s_nn_new = np.zeros(3 ** n)
                        for j in range(3):
                            for k in range(len(vec_c_nn_old)):
                                const = (2*n-1)/math.sqrt(2*n*(2*n+1))
                                vec_c_nn_new[j*3**(n-1)+k] = (x_dot[j]*vec_c_nn_old[k]/a - y_dot[j]*vec_s_nn_old[k]/a)*const
                                vec_s_nn_new[j*3**(n-1)+k] = (y_dot[j]*vec_c_nn_old[k]/a + x_dot[j]*vec_s_nn_old[k]/a)*const
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

    C_nm[0] = density / mass

    if nmax > 0:
        C_nm[2] = density / mass
        C_nm[1] = density / mass
        S_nm[2] = density / mass

    if nmax > 1:
        count = 3
        for n in range(2, nmax+1):
            for m in range(0, n+1):
                C_nm[count] = density/mass
                if m == 0:
                    S_nm[count] = 0
                else:
                    S_nm[count] = density / mass
                count += 1

    C_nm *= sum_c
    S_nm *= sum_s
    return C_nm, S_nm
