import numpy as np
from dataset.mnist import load_mnist
from my_two_layer_net import TwoLayerNet
import matplotlib.pylab as plt


batch_size_df = 100  # size of training data
learning_times_df = 10000  # learning times
learning_rate_df = 0.1  # learning rate
hidden_size_df = 50
output_size_df = 10


def training(batch_size=batch_size_df, learning_times=learning_times_df, learning_rate=learning_rate_df,
             hidden_size=hidden_size_df, output_size=output_size_df):
    print('> Load training data')
    (x_train, t_train), (_x_test, _t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_size, data_size = x_train.shape[0], len(x_train[0])
    print('training data loading Complete, parameters are')
    print('dataset size: {}\none data point size: {}\nbatch size: {}\n'.format(train_size, data_size, batch_size))
    print('> Start learning')
    print('learning times: {}\nlearning rate: {}'.format(learning_times, learning_rate))
    print('network hidden size: {}\noutput size: {}\n'.format(hidden_size, output_size))
    network = TwoLayerNet(input_size=data_size, hidden_size=hidden_size, output_size=output_size)
    for i in range(learning_times):
        print('learning #{}'.format(i))
        # get mini-batch
        batch_mask = np.random.choice(train_size, batch_size)  # [5, 0, 12, 99, 6, ...]
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # get grads
        grad = network.numerical_gradient(x_batch, t_batch)

        # update params
        for k in ('w1', 'b1', 'w2', 'b2'):
            network.params[k] -= learning_rate * grad[k]

    print('learning complete')
    plt.plot(network.loss_record)
    plt.show()


if __name__ == '__main__':
    training()