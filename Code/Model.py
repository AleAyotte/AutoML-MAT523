"""
    @file:              DataGen.py
    @Author:            Nicolas Raymond
    @Creation Date:     30/09/2019
    @Last modification: 30/09/2019
    @Description:       This program generate models of different types to use for our classification problems.

"""

import numpy as np

class Model:

    def __init__(self, HP_Dict):

        """
        Class that generate a model to solve a classification problem

        :param HP_Dict: Dictionary containing all hyperparameters of the model

        """

        Hyperparam = HP_Dict


    def train(self):

        """
        Train our model

        Note that it wil be override by the children's classes
        """

        return NotImplementedError


    def predict(self, x):

        """
        Predict the class for an observation x

        :param x: Value (float) or 1-d numpy array of values
        :return: Class (integer)
        """

        return NotImplementedError

    def accuracy(self, X, T):

        """

        :param X: NxD numpy array of observations (N : nb of obs, D : nb of dimensions)
        :param T: Nx1 numpy array
        :return: Good classification rate

        """
        predictions = np.array([self.predict(x) for x in X]) # Nx1 array containing predictions
        diff = T - predictions

        return ((diff == 0).sum())/len(diff) # (Nb of good predictions / nb of predictions)

