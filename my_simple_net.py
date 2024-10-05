import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:

    def __init__(self, x, t):
        # x > training data; t > expectation
        self.x = x
        self.t = t
        self.w = np.random.randn(2, 3)

    def predict(self):
        return np.dot(self.x, self.w)

    def loss(self):
        z = self.predict()
        y = softmax(z)
        return cross_entropy_error(y, t)


if __name__ == '__main__':
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    net = SimpleNet(x, t)
    def f(_):
        return net.loss()

    dw = numerical_gradient(f, net.w)
    print(dw)
