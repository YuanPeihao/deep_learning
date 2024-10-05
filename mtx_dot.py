import numpy as np


if __name__ == '__main__':
    mtx_a = np.array([[1, 2], [3, 4], [5, 6]])
    mtx_b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    mtx_c = np.dot(mtx_a, mtx_b)
    print(mtx_c)