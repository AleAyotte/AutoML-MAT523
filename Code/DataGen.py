"""
    @file:              DataGen.py
    @Author:            Alexandre Ayotte
    @Creation Date:     29/09/2019
    @Last modification: 29/09/2019
    @Description:       This program generate random data for toy problem that we want to solve with autoML method
"""

from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
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

        else:
            print("Model: {} does not exist in this program".format(self.model))
            return -1   # Return -1 cause the

        return x_train, t_train, x_test, t_test
