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
        HP_space = HP_Dict


    def train(self, X_train, t_train):

        """
        Train our model

        Note that it wil be override by the children classes

        :param X_train: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observation

        """

        return NotImplementedError


    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array of classes predicted for each observation
        """

        return NotImplementedError

    def accuracy(self, X, t):

        """
        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :return: Good classification rate

        """
        predictions = self.predict(X)
        diff = t - predictions

        return ((diff == 0).sum())/len(diff) # (Nb of good predictions / nb of predictions)



class SVM(Model):

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma='auto_deprecated', coef0=0.0, max_iter=-1):

        """
        Class that generates a support vector machine

        Take a look at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC for
        more informations on hyperparameters

        :param C: Penalty parameter C of the error term
        :param kernel: Kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
        :param degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
        :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        :param coef0: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        """

        model = sk.svm.SVC(C, kernel, degree, gamma, coef0, max_iter=max_iter)
        super().__init__({'C':C, 'kernel':kernel, 'degree':degree, 'gamma':gamma, 'coef0':coef0})


    def train(self, X_train, t_train):

        """
        Train our model

        :param X_train: NxD numpy array of the observations of the training set {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observation in the training set
        """

        self.model.fit(X_train, t_train)


    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array of classes predicted for each observation

        """

        return self.model.predict(X)


class MLP(Model):

    def __init__(self, hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9, beta_1=0.9,
                 beta_2=0.999):

        """

        Class that generate a Multi-layer Perceptron classifier.

        This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

        Some hyperparameters are conditional to others!
        For more informations take a look at :
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

        :param hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer
        :param activation: Activation function for the hidden layer {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        :param solver: The solver for weight optimization {‘lbfgs’, ‘sgd’, ‘adam’}
        :param alpha: L2 penalty (regularization term) parameter
        :param batch_size: Size of minibatches for stochastic optimizers.
        :param learning_rate: Learning rate schedule for weight updates {‘constant’, ‘invscaling’, ‘adaptive’}
        :param learning_rate_init: The initial learning rate used
        :param power_t: The exponent for inverse scaling learning rate.
        :param max_iter: Maximum number of iterations
        :param momentum: Momentum for gradient descent update
        :param beta_1: Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1)
        :param beta_2: Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1)
        """

        model = sk.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                                                alpha=0.0001, batch_size='auto',learning_rate='constant', learning_rate_init=0.001,
                                                power_t=0.5, max_iter=200, momentum=0.9, beta_1=0.9, beta_2=0.999)

        super().__init__({'hidden_layer_sizes':hidden_layer_sizes, 'activation':activation, 'solver':solver, 'alpha':alpha,
                          'batch_size':batch_size, 'learning_rate':learning_rate, 'power_t':power_t, 'max_iter':max_iter,
                          'momentum':momentum, 'beta_1':beta_1, 'beta_2':beta_2})

        def train(self, X_train, t_train):

            """
            Train our model

            :param X_train: NxD numpy array of the observations of the training set
            :param t_train: Nx1 numpy array classes associated with each observations in the training set
            """

            self.model.fit(X_train, t_train)


        def predict(self, X):

            """
            Predict classes for our observations in the input array X

            :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
            :return: Nx1 numpy array classes predicted for each observation

            """

            return self.model.predict(X)

