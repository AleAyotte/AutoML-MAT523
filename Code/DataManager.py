"""
    @file:              DataGen.py
    @Author:            Alexandre Ayotte
    @Creation Date:     29/09/2019
    @Last modification: 29/09/2019
    @Description:       This program generates randoms data for toy problems that we want to solve with autoML methods
"""

from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
import numpy as np
import math
import copy
import matplotlib.pyplot as plt


class DataGenerator:

    def __init__(self, train, test, model_name):
        """
        Class that generates training and testing samples from sklearn datasets.

        :param train: The training sample length
        :param test: The testing sample length
        :param model_name: The name of the model that will be used to generate the sample.
        """
        self.train_size = train
        self.test_size = test
        self.model = model_name

    @staticmethod
    def polar_to_cart(radius, angle):
        """
        Convert coordinates from polar to cartesian coordinates

        :param radius: A numpy vector of size N that represents the radius of each data points
        :param angle: A numpy vector of size N that represents the angle of each data points
        :return: Two numpy vector of size N that represents the cartesian coordinates
        """

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return x, y

    def n_spiral(self, nb_data, noise, num_class):
        """

        :param nb_data: Number of data points that we want to generates
        :param noise: The standard deviation of the Gaussian noise added to the data
        :param num_class: The number of classes
        :return: A Nx2 numpy matrix that contains the data points coordinates and a array of size N that represents the
                labels of the data points
        """
        labels = np.array([])
        radius = 3 * math.pi*np.random.rand(nb_data)
        angle = np.array([])

        data_left = nb_data
        angle_step = 2 * math.pi / num_class

        for it in range(num_class):
            split = round(data_left / (num_class - it))
            temp = angle_step*it + np.random.normal(size=split, loc=0, scale=noise)

            data_left -= split
            angle = np.append(angle, temp)
            labels = np.append(labels, np.ones(split) * it)

        angle = angle + copy.copy(radius)
        radius += 1
        features = np.vstack([self.polar_to_cart(radius, angle)]).T
        return features, labels.astype(dtype='int32')

    def generate_data(self, noise=0, num_class=2):
        """
        Generates random training and testing sample according to the model name

        :param noise: The standard deviation of the Gaussian noise added to the data
        :param num_class: Number of classes, only for the nSpiral model
        :return: 4 numpy array for training features, training labels, testing features and testing labels respectively
        """
        if self.model == "half_moon":
            x_train, t_train = make_moons(self.train_size, noise=noise)
            x_test, t_test = make_moons(self.test_size, noise=noise)

        elif self.model == "circles":
            x_train, t_train = make_circles(self.train_size, noise=noise)
            x_test, t_test = make_circles(self.test_size, noise=noise)

        elif self.model == "nSpiral":
            x_train, t_train = self.n_spiral(self.train_size, noise=noise, num_class=num_class)
            x_test, t_test = self.n_spiral(self.test_size, noise=noise, num_class=num_class)

        # else this model doesn't exist in this program and we want to throw an error message
        else:
            raise Exception("Model: {} does not exist in this program".format(self.model))

        return x_train, t_train, x_test, t_test


def create_dataloader(features, labels, b_size):
    """
    Transform a n dimensional numpy array of features and a numpy array of labels into ONE data loader

    :param features: A n dimensional numpy array that contain features of each data points
    :param labels: A numpy array that represent the correspondent labels of each data points
    :param b_size: Batch size as integer
    :return: A dataloader that contain
    """
    tensor_x = torch.tensor(features, dtype=torch.float)
    tensor_y = torch.tensor(labels, dtype=torch.long)
    dt = utils.TensorDataset(tensor_x, tensor_y)
    dt_load = utils.DataLoader(dt, batch_size=b_size, shuffle=True, drop_last=True)

    return dt_load


def dataset_to_loader(dataset, b_size=12, shuffle=False):
    """
    Transform a torch dataset into a torch dataloader who provide an iterable over the dataset

    :param dataset: A torch dataset
    :param b_size: The batch size
    :param shuffle: If the dataset is shuffle at each epoch
    :return: A torch data_loader that contain the features and the labels.
    """
    data_loader = utils.DataLoader(dataset, batch_size=b_size, shuffle=shuffle, drop_last=True)

    return data_loader


def load_cifar10():
    """
    Load the CIFAR10 dataset using pytorch
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :return: The train set and the test set of the CIFAR10 dataset as pytorch Dataset
    """

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return trainset, testset


def load_mnist():
    """
    Load the MNIST dataset using pytorch
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :return: The train set and the test set of the CIFAR10 dataset as pytorch Dataset
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=Test, download=True, transform=transform)

    return trainset, testset


def plot_data(data, target):
    """
    Show a two dimensional dataset in a chart.

    :param data: A numpy array of dimension Nx2, that represents the coordinates of each data points.
    :param target: A numpy array of integer value that represents the data points labels
    """
    plt.scatter(data[:, 0], data[:, 1], s=105, c=target, edgecolors='b')
    plt.show()
