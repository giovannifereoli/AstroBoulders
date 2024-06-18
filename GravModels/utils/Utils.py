import numpy as np
from GravModels.utils.ProgressBar import ProgressBar


def compute_acceleration_grid(num, feature_x, feature_y, feature_z, model_func):
    acc_vec_x = np.zeros((num, num, 3))
    acc_vec_y = np.zeros((num, num, 3))
    acc_vec_z = np.zeros((num, num, 3))

    bar = ProgressBar(num)
    for i in range(num):
        for j in range(num):
            position_x = np.array([[0, feature_y[i], feature_z[j]]])
            position_y = np.array([[feature_x[i], 0, feature_z[j]]])
            position_z = np.array([[feature_x[i], feature_y[j], 0]])

            acc_vec_x[i, j] = model_func(position_x)
            acc_vec_y[i, j] = model_func(position_y)
            acc_vec_z[i, j] = model_func(position_z)
        bar.update(i)
    bar.markComplete()
    bar.close()
    return acc_vec_x, acc_vec_y, acc_vec_z
