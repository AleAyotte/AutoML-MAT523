"""
    @file:              DataGen.py
    @Author:            Nicolas Raymond
    @Creation Date:     30/09/2019
    @Last modification: 30/09/2019
    @Description:       This program generate models of different types to use for our classification problems.

"""

import numpy as np
import sklearn as sk

class Model:

    def __init__(self, HP_Dict):

        """
        Class that generate a model to solve a classification problem

        :param HP_Dict: Dictionary containing all hyperparameters of the model

        """
        hyperparams = HP_Dict


    def train(self, X_train, t_train):

        """
        Train our model

        Note that it wil be override by the children's classes

        :param X_train: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observations

        """

        return NotImplementedError


    def predict(self, X):

        """
        Predict the class for an observation x

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array of classes predicted for each observation
        """

        return NotImplementedError

    def accuracy(self, X, T):

        """
        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param T: Nx1 numpy array of classes associated with each observation
        :return: Good classification rate

        """
        predictions = self.predict(X)
        diff = T - predictions

        return ((diff == 0).sum())/len(diff) # (Nb of good predictions / nb of predictions)


class SVM(Model):

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma='auto_deprecated', coef0=0.0):

        """
        Class that generate a support vector machine

        Take a look at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC for
        more informations on hyperparameters

        :param C: Penalty parameter C of the error term
        :param kernel: Kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
        :param degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
        :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        :param coef0: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        """

        model = sk.svm.SVC(C, kernel, degree, gamma, coef0)
        super().__init__({'C':C, 'kernel':kernel, 'degree':degree, 'gamma':gamma, 'coef0':coef0})


    def train(self, X_train, t_train):

        """
        Train our model

        :param X_train: NxD numpy array of the observations of the training set
        :param t_train: Nx1 numpy array classes associated with each observations in the training set
        """

        self.model.fit(X_train, t_train)


    def predict(self, X):

        """
        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array classes predicted for each observation

        """

        return self.model.predict(X)

