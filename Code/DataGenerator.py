"""
 *  DataGenerator.py
 *  @authors: Alexandre Ayotte and Nicolas Raymond
 *  @version:   0.01 - 2019-09-10
 *
 *  Description:
     
 *   The goal of this file is to provide a class DataGenerator that we'll use to generate
 *   training and testing data sets of data that will be helpful to test performance of different autoML tools.
 *   For example, N-Circle and N-Region.

"""

import numpy as np
import math
import matplotlib.pyplot as plt


class DataGenerator:
    
    def __init__(self, nb_train, nb_test, nb_class):
        
        """
        Class that generates training and testing datasets of different kinds.

        :param nb_train: Number of training data points
        :param nb_test:  Number of training data points
        :param nb_class: Number of classes
        """
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.nb_class = nb_class

    @staticmethod
    def polar_to_cart(radius, angle):
        
        """
        Converts polar coordinates into cartesian coordinates

        :param radius: numpy array containing radius of every points in our 2D space
        :param angle: numpy array containing angles of every points in our 2D space
        :return: two array containing x and y coordinates respectively
        """

        axis_x = radius * np.cos(angle)
        axis_y = radius * np.sin(angle)
        return axis_x.astype(dtype='float32'), axis_y.astype(dtype='float32')

    def n_circle(self, nb_data, noise, distance):
        
        """
        Generates random datasets of type N-Circle where N represents number of classes.
        Data points are normally distributed in every classes using Gaussian distribution
        over radius.

        :param nb_data: Number of data points to generate
        :param noise: Standard deviation use in our Gaussian distribution
        :param distance: Expectation over radius difference between each classes (circles)
        :return: Two numpy array containing cartesian coordinates and class targets respectively
        """

        classes = np.array([])
        radius = np.array([])

        angle = 2*math.pi*np.random.rand(nb_data)

        # Number of data points left to generate
        data_left = nb_data

        for i in range(self.nb_class):
            
            # We split data points fairly between classes (circles)
            split = round(data_left/ (self.nb_class - i))

            # We generate a numpy array containing normally distributed value of radius
            temp = np.random.normal(size=split, loc=distance*(i + 1), scale=noise)
            data_left -= split
            
            # We update our radius and classes arrays
            radius = np.append(radius, temp)
            classes = np.append(classes, np.ones(split) * i)
        
        # We convert our data points to cartesian coordinates
        data = np.vstack([self.polar_to_cart(radius, angle)]).T
        
        return data, classes.astype(dtype='int32')

    def n_region(self, nb_data, noise):
        
        """
        Generates random datasets of type N-Region where N represents number of classes. 
        Every region should look like a triangle.

        :param nb_data: Number of data points to generate.
        :param noise: Standard deviation use for our Gaussian distribution over angles
        :return: Two numpy array containing cartesian coordinates and class targets respectively
        """

        classes = np.array([])
        angle = np.array([])

        radius = 4 * np.random.rand(nb_data) + 1
        
        # Number of data points left to generate
        data_left = nb_data

        for i in range(self.nb_class):
            
            # We split data points fairly between classes (regions)
            split = round(data_left / (self.nb_class - i))

            # We generate a numpy array containing normally distributed value of angles
            temp = np.random.normal(size=split, loc=(2*math.pi*i /self.nb_class), scale=noise)
            data_left -= split
            
            # We update our angle array
            angle = np.append(angle, temp)
            classes = np.append(classes, np.ones(split) * i)
        
        # We convert our data points to cartesian coordinates
        data = np.vstack([self.polar_to_cart(radius, angle)]).T
        
        return data, classes.astype(dtype='int32')

    def n_spike(self, nb_data, noise, distance, amplitude):
        
        """
        Generates random datasets of type N-Spike where N represents number of classes.
        Every region should look like a circle with spike
    
        :param nb_data: Number of data points to generate.
        :param noise: Standard deviation use for our Gaussian distribution over distance from origin
        :param distance: Mean distance between each classes
        :param amplitude: Sinusoidal amplitude.
        :return: Two numpy array containing cartesian coordinates and class targets respectively.
        """

        classes = np.array([])
        radius = np.array([])

        angle = 2 * math.pi * np.random.rand(nb_data)

        # Number of data points left to generate
        data_left = nb_data

        for i in range(self.nb_class):
            # We split data points fairly between classes
            split = round(data_left / (self.nb_class - i))

            # We calculate the distance from the origin
            mean = distance * (i + 1)

            # We generate a vector of random radius for this class
            temp = np.random.normal(size=split, loc=mean, scale=noise)
            data_left -= split
            radius = np.append(radius, temp)
            classes = np.append(classes, np.ones(split) * i)
            np.set_printoptions(precision=3)

        radius += amplitude * np.sin(5*angle)
        data = np.vstack([self.polar_to_cart(radius, angle)]).T
        return data, classes.astype(dtype='int32')

    def generate_data(self, modele='nCircle', noise=0.5, distance=5, amplitude=1):

        """
        Generate training and test dataset according the given model.

        :param modele: The function that will be use to generate the data. Example: nCircle, nSpike...
        :param noise:  The standard deviation of the normal distribution that generate the data
        :param distance:  The distance between each ring in the nCircle model
        :param amplitude:   Sinusoidal amplitude of the nSpike model
        :return:    2 numpy matrix that contain training features and test features and 2 numpy array of
                    training and test labels.
        """

        if modele == "nCircle":
            x_train, t_train = self.n_circle(self.nb_train, noise, distance)
            x_test, t_test = self.n_circle(self.nb_test, noise, distance)
        elif modele == "nRegion":
            x_train, t_train = self.n_region(self.nb_train, noise)
            x_test, t_test = self.n_region(self.nb_test, noise)
        elif modele == "nSpike":
            x_train, t_train = self.n_spike(self.nb_train, noise, distance, amplitude)
            x_test, t_test = self.n_spike(self.nb_test, noise, distance, amplitude)
        else:
            return -1
        return x_train, t_train, x_test, t_test


def show_data_points(data, classes):

    """
    Allows to show our data points in a graphic

    :param data: A Nx2 matrix that represents the data points coordinates
    :param classes: An array that contain the label of each data point
    :return:
    """

    plt.scatter(data[:, 0], data[:, 1], s=105, c=classes, edgecolors='b')
    plt.show()