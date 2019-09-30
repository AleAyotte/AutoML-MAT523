"""
    @file:              DataGen.py
    @Author:            Alexandre Ayotte
    @Creation Date:     29/09/2019
    @Last modification: 29/09/2019
    @Description:       This program generate random data for toy problem that we want to solve with autoML method
"""

from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_swiss_roll
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:

    def __init__(self, train, test, model_name):
        """
        Class that generate training and testing sample from sklearn datasets.

        :param train: The training sample length
        :param test: The testing sample length
        :param model_name: The name of the model that will be use to generate the samples.
        """
        self.train_size = train
        self.test_size = test
        self.model = model_name

    def generate_data(self, noise):
        """
        Generate random training and testing sample according to the model name

        :param noise: The standard deviation of the Gaussian noise added to the data
        :return: 4 numpy array for training features, training labels, testing features and testing labels respectively
        """
        if self.model == "half_moon":
            x_train, t_train = make_moons(self.train_size, noise=noise)
            x_test, t_test = make_moons(self.test_size, noise=noise)

        elif self.model == "circles":
            x_train, t_train = make_circles(self.train_size, noise=noise)
            x_test, t_test = make_circles(self.test_size, noise=noise)

        elif self.model == "swiss_roll":
            x_train, t_train = make_swiss_roll(self.train_size, noise=noise)
            x_test, t_test = make_swiss_roll(self.test_size, noise=noise)

        # else this model doesn't exist in this program and we want to throw an error message
        else:
            print("Model: {} does not exist in this program".format(self.model))
            return -1

        return x_train, t_train, x_test, t_test


def plot_data(data, target):
    """
    Show a two dimensional dataset in a chart.

    :param data: A numpy array of dimension Nx2, that represent the coordinates of each data point.
    :param target: The sample labels
    """
    label_0 = np.where(target == 0, True, False)
    data_0 = data[label_0, :]
    data_1 = data[~label_0, :]

    plt.plot(data_0[:, 0], data_0[:, 1], 'bs', data_1[:, 0], data_1[:, 1], 'g^')
    plt.show()
