import numpy as np
import matplotlib.pylab as plt


# Loss functions
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))


def batch_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# Derivative functions
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx, val in enumerate(x):  # take the derivative for each element in a matrix
        x[idx] = val + h
        fxh1 = f(x)
        x[idx] = val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


if __name__ == '__main__':
    # x = np.arange(0.0, 20.0, 0.1)
    # y = function_1(x)
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.plot(x, y)
    # plt.show()
    # d5 = numerical_diff(function_1, 5)
    # d10 = numerical_diff(function_1, 10)
    # print(d5, d10)
    # gradient at (3, 4) (0, 2) and (3, 0)
    # r1 = numerical_gradient(function_2, np.array([3.0, 4.0]))
    # r2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
    # r3 = numerical_gradient(function_2, np.array([3.0, 0.0]))
    # print(r1, r2, r3)
    init_x = np.array([-3.0, 4.0])
    f_min = gradient_descent(f=function_2, init_x=init_x, lr=0.1, step_num=100)
    print(f_min)



