import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from PIL import Image

from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def img_show(img: list):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_test_data():
    _, (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network: dict, x: list):
    # print('Predicting')
    w = [network['W1'], network['W2'], network['W3']]
    b = [network['b1'], network['b2'], network['b3']]
    z = x
    for layer in range(3):
        print('Layer {}'.format(layer))
        cur_w, cur_b = w[layer], b[layer]
        a = np.dot(z, cur_w) + cur_b
        print('a is {}'.format(a))
        # print('input shape {}'.format(z.shape))
        # print('weight shape {}'.format(cur_w.shape))
        # print('output shape {}'.format(a.shape))
        if layer == 2:
            y = softmax(a)
            print('softmax y is {}'.format(y))
            return y

        z = sigmoid(a)
        print('sigmoid z is {}'.format(z))


if __name__ == '__main__':
    # (train image, train label), (test image, test label)
    # (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    # print(x_train[0])
    # img, label = x_train[0], t_train[0]
    # img = img.reshape(28, 28)
    # print(label)
    # img_show(img)
    x, t = get_test_data()
    nw = init_network()
    y = predict(nw, x[0])
    # batch_size = 100
    # accuracy_cnt = 0
    # for i in range(0, len(x), batch_size):
    #     x_batch = x[i: i+batch_size]
    #     y_batch = predict(nw, x_batch)
    #     p = np.argmax(y_batch, axis=1)
    #     accuracy_cnt += np.sum(p == t[i: i+batch_size])
    #
    # print('Accuracy: {}%'.format(100*(float(accuracy_cnt)/len(x))))





