import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            'w1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'w2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
        self.loss_record = list()

    def predict(self, x):
        w1, w2 = self.params.get('w1'), self.params.get('w2')
        b1, b2 = self.params.get('b1'), self.params.get('b2')
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        return softmax(a2)

    def loss(self, x, t):
        y = self.predict(x)
        err = cross_entropy_error(y, t)
        self.loss_record.append(err)
        return err

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {
            'w1': numerical_gradient(loss_w, self.params['w1']),
            'b1': numerical_gradient(loss_w, self.params['b1']),
            'w2': numerical_gradient(loss_w, self.params['w2']),
            'b2': numerical_gradient(loss_w, self.params['b2'])
        }
        return grads
    