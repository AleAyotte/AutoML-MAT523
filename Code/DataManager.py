"""
    @file:              DataGen.py
    @Author:            Alexandre Ayotte
    @Creation Date:     29/09/2019
    @Last modification: 17/11/2019
    @Description:       This program generates randoms data  for toy problems, load dataset from torchvision and csv
                        and provide useful method to manipulate data.

"""

from sklearn.datasets import make_moons, make_circles, load_iris, fetch_covtype, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils


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
        Generates random training and testing sample with spiral shapes.

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

    def generate_data(self, noise=0, num_class=2, seed=None):

        """
        Generates random training and testing sample according to the model name

        :param noise: The standard deviation of the Gaussian noise added to the data
        :param num_class: Number of classes, only for the nSpiral model
        :param seed: Set the seed of the numpy random state
        :return: 4 numpy arrays for training features, training labels, testing features and testing labels respectively
        """

        np.random.seed(seed=seed)

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

        # restore the numpy seed
        np.random.seed(seed=None)

        return x_train, t_train, x_test, t_test


def create_dataloader(features, labels, b_size, shuffle=False):

    """
    Transforms a n dimensional numpy array of features and a numpy array of labels into ONE data loader

    :param features: A n dimensional numpy array that contain features of each data points
    :param labels: A numpy array that represent the correspondent labels of each data points
    :param b_size: Batch size as integer
    :param shuffle: Do we want to shuffle the data at each epoch
    :return: A dataloader that contain given features and labels.
    """
    tensor_x = torch.tensor(features, dtype=torch.float)
    tensor_y = torch.tensor(labels, dtype=torch.long)
    dt = utils.TensorDataset(tensor_x, tensor_y)
    dt_load = utils.DataLoader(dt, batch_size=b_size, shuffle=shuffle, drop_last=True)

    return dt_load


def dataset_to_loader(dataset, b_size=12, shuffle=False):

    """
    Transforms a torch dataset into a torch dataloader who provide an iterable over the dataset

    :param dataset: A torch dataset
    :param b_size: The batch size
    :param shuffle: If the dataset is shuffle at each epoch
    :return: A torch data_loader that contain the features and the labels.
    """
    data_loader = utils.DataLoader(dataset, batch_size=b_size, shuffle=shuffle, drop_last=True)

    return data_loader


def validation_split(features=None, labels=None, dtset=None, valid_size=0.2, random_state=None):

    """
    Splits a torch dataset or features and labels numpy arrays into two dataset or two features and two labels numpy
    arrays respectively.

    :param features: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
    :param labels: Nx1 numpy array of classes associated with each observation
    :param dtset: A torch dataset which contain our train data points and labels
    :param valid_size: Proportion of the dataset that will be use as validation data
    :param random_state: Seed used by the random number generator
    :return: train and valid features as numpy arrays and train and valid labels as numpy arrays if features and labels
             numpy arrays are given but no torch dataset. Train and valid torch datasets if a torch dataset is given.
    """

    if dtset is None:
        if features is None or labels is None:
            raise Exception("Features or labels missing. X is None: {}, t is None: {}, dtset is None: {}".format(
                features is None, labels is None, dtset is None))
        else:
            # x_train, x_valid, t_train, t_valid
            return train_test_split(features, labels, test_size=valid_size, random_state=random_state)

    else:
        num_data = len(dtset)
        num_valid = math.floor(num_data * valid_size)
        num_train = num_data - num_valid

        # d_train, d_valid
        return utils.dataset.random_split(dtset, [num_train, num_valid])


def load_cifar10():

    """
    Loads the CIFAR10 dataset using pytorch
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :return: The train set and the test set of the CIFAR10 dataset as pytorch Dataset
    """

    # Data augmentation for training
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # For the test set, we just want to normalize it
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def load_cifar100():

    """
    Loads the CIFAR100 dataset using pytorch
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :return: The train set and the test set of the CIFAR100 dataset as pytorch Dataset
    """

    # Data augmentation for training
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # For the test set, we just want to normalize it
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def load_svhn():

    """
    Loads the SVHN dataset using pytorch
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :return: The train set and the test set of the CIFAR10 dataset as pytorch Dataset
    """

    # Data augmentation for training
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize((111.6089, 113.1613, 120.5651),
                                                               (50.4977, 51.2590, 50.2442))])

    # For the test set, we just want to normalize it
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((111.6089, 113.1613, 120.5651),
                                                              (50.4977, 51.2590, 50.2442))])

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)

    return trainset, testset


def load_stl10():

    """
    Loads the SVHN dataset using pytorch
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :return: The train set and the test set of the CIFAR10 dataset as pytorch Dataset
    """

    # Data augmentation for training
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    # For the test set, we just want to normalize it
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

    return trainset, testset


def load_mnist():

    """
    Loads the MNIST dataset using pytorch and normalize it using is
    inspired by pytorch tutorial "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"

    :return: The train set and the test set of the MNIST dataset as pytorch Dataset
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return trainset, testset


def load_csv(path, label_col, test_split=0.2):

    """
    Loads a dataset from a csv file

    :param path: The path of the csv file.
    :param label_col: The number of the column that contain the labels. (First column is 0)
    :param test_split: Proportion of the dataset that will be use as test data (Default 0.2 = 20%)
    :return: train and valid features as numpy arrays and train and valid labels as numpy arrays
    """
    data_csv = pd.read_csv(path).values

    features = np.delete(data_csv, label_col, axis=1)
    labels = data_csv[:, label_col]

    # Scaling features
    scaler = preprocessing.StandardScaler()
    scaler.fit(features)
    normalized_features = scaler.transform(features)

    # Encode the labels
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    encoded_labels = le.transform(labels)

    if test_split > 0:
        return validation_split(normalized_features, encoded_labels, valid_size=test_split)

    else:
        return normalized_features, encoded_labels


def load_iris_dataset(scaled=True, test_split=0.2, random_state=None):

    """
    Load iris classification dataset offered by sklearn
    https://scikit-learn.org/stable/datasets/index.html#iris-dataset

    :return: 4 numpy arrays for training features, training labels, testing features and testing labels respectively
    """
    data = load_iris()
    X = data['data']

    if scaled:
        X = preprocessing.scale(X)

    t = data['target']

    x_train, x_test, t_train, t_test = validation_split(X, t, valid_size=test_split, random_state=random_state)

    return x_train, t_train, x_test, t_test


def load_forest_covertypes_dataset(test_split=0.2, random_state=None):

    """
    Loads forest covertypes dataset offered by sklearn
    https://scikit-learn.org/stable/datasets/index.html#forest-covertypes

    :param test_split: test_split: Proportion of the dataset that will be use as test data (Default 0.2 = 20%).
    :param random_state: Seed generator that will be uses for splitting the data.
    :return: 4 numpy arrays for training features, training labels, testing features and testing labels respectively
    """
    data = fetch_covtype()
    X = data['data']
    t = data['target']

    x_train, x_test, t_train, t_test = validation_split(X, t, valid_size=test_split, random_state=random_state)

    return x_train, t_train, x_test, t_test


def load_breast_cancer_dataset(scaled=True, test_split=0.2, random_state=None):

    """
    Loads the breast cancer wisconsin dataset for classication task

    :param scaled: True for scaling the data.
    :param test_split: test_split: Proportion of the dataset that will be use as test data (Default 0.2 = 20%).
    :param random_state: Seed generator that will be uses for splitting the data.
    :return: 4 numpy arrays for training features, training labels, testing features and testing labels respectively.
    """

    data, target = load_breast_cancer(True)

    if scaled:
        data = preprocessing.scale(data)

    x_train, x_test, t_train, t_test = validation_split(data, target, valid_size=test_split, random_state=random_state)
    return x_train, t_train, x_test, t_test


def load_digits_dataset(test_split=0.2):

    """
    Load hand written digits classification dataset provided by sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits

    :param test_split: test_split: Proportion of the dataset that will be use as test data (Default 0.2 = 20%).
    :return: 4 numpy arrays for training features, training labels, testing features and testing labels respectively.
    """
    # The digits dataset
    digits = load_digits()

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Split data into train and test subsets
    x_train, x_test, t_train, t_test = validation_split(data, digits.target, valid_size=test_split)

    return x_train, t_train, x_test, t_test


def plot_data(data, target):

    """
    Shows a two dimensional dataset in a chart.

    :param data: A numpy array of dimension Nx2, that represents the coordinates of each data points.
    :param target: A numpy array of integer value that represents the data points labels
    """
    plt.scatter(data[:, 0], data[:, 1], s=105, c=target, edgecolors='b')
    plt.show()
