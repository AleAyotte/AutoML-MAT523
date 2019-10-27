"""
    @file:              Model.py
    @Author:            Nicolas Raymond
    @Creation Date:     30/09/2019
    @Last modification: 08/10/2019
    @Description:       This program generates models of different types to use for our classification problems.

"""

import numpy as np
import sklearn.svm as svm
import sklearn.neural_network as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from enum import Enum, unique


@unique
class HPtype(Enum):

    """
    Class containing possible types of hyper-parameters

    """
    DISCRETE = 1
    CONTINUOUS = 2
    CATEGORICAL = 3


class Hyperparameter:

    def __init__(self, name, type, value=None):
        """

        Class that defines an hyper-parameter

        :param name: Name of the hyper-parameter
        :param type: One type out of HPtype (DISCRETE, CONTINUOUS, CATEGORICAL)
        :param value: List with the value of the hyper-parameter

        """
        self.name = name
        self.type = type
        self.value = value

        if self.type == HPtype.DISCRETE:
            self.type_name = 'discrete'

        elif self.type == HPtype.CONTINUOUS:
            self.type_name = 'continuous'

        else:
            self.type_name = 'categorical'


class Model:

    def __init__(self, HP_Dict):

        """
        Class that generates a model to solve a classification problem

        :param HP_Dict: Dictionary containing all hyper-parameters of the model

        """
        self.HP_space = HP_Dict

    def set_hyperparameters(self, hyperparams):

        """
        Change hyper-parameters of our model

        :param hyperparams: Dictionary of hyper-parameters to change

        """

        raise NotImplementedError

    def fit(self, X_train, t_train):

        """
        Train our model

        Note that it will be override by the children's classes

        :param X_train: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observation

        """

        raise NotImplementedError

    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array of classes predicted for each observation
        """

        raise NotImplementedError

    def score(self, X, t):

        """
        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :return: Good classification rate

        """
        predictions = self.predict(X)

        diff = t - predictions

        return ((diff == 0).sum())/len(diff)  # (Nb of good predictions / nb of predictions)

    def cross_validation(self, X_train, t_train, nb_of_cross_validation=3):

        """

        :param X_train: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observation
        :param nb_of_cross_validation:  Number of data splits and validation to execute
        :return: Mean of score (accuracy)

        """
        res = np.array([])

        for i in range(nb_of_cross_validation):

            x_train, x_test, y_train, y_test = train_test_split(X_train, t_train, test_size=0.2)
            self.fit(x_train, y_train)
            res = np.append(res, self.score(x_test, y_test))

        return np.mean(res)

    def plot_data(self, data, classes):

        """

        Plot data points and spaces of separation done by the model for 2D cases only.

        :param data: Nx2 numpy array of observations {N : nb of obs}
        :param classes: Classes associate with each data point.
        """

        if data.shape[1] != 2:
            raise Exception('Method only available for 2D plotting (two dimensions datasets)')

        else:
            ix = np.arange(data[:, 0].min(), data[:, 0].max(), 0.01)
            iy = np.arange(data[:, 1].min(), data[:, 1].max(), 0.01)
            iX, iY = np.meshgrid(ix, iy)
            x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
            contour_out = self.predict(x_vis)
            contour_out = contour_out.reshape(iX.shape)

            plt.contourf(iX, iY, contour_out)
            plt.scatter(data[:, 0], data[:, 1], s=105, c=classes, edgecolors='b')
            plt.title('Accuracy on test : {} %'.format(self.score(data, classes)*100))
            plt.show()


class SVM(Model):

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma='auto', coef0=0.0, max_iter=-1):

        """
        Class that generates a support vector machine

        Some hyper-parameters are conditional to others!
        Take a look at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC for
        more information on hyper-parameters

        :param C: Penalty parameter C of the error term
        :param kernel: Kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’ or ‘sigmoid’
        :param degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
        :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        :param coef0: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        """

        self.model_frame = svm.SVC(C, kernel, degree, gamma, coef0, max_iter=max_iter)

        if kernel == 'rbf':
            super(SVM, self).__init__({'C': Hyperparameter('C', HPtype.CONTINUOUS, [C]),
                                       'kernel': Hyperparameter('kernel', HPtype.CATEGORICAL, [kernel]),
                                       'gamma': Hyperparameter('gamma', HPtype.CONTINUOUS, [gamma])})

        elif kernel == 'linear':
            super(SVM, self).__init__({'C': Hyperparameter('C', HPtype.CONTINUOUS, [C]),
                                       'kernel': Hyperparameter('kernel', HPtype.CATEGORICAL, [kernel])})

        elif kernel == 'poly':
            super(SVM, self).__init__({'C': Hyperparameter('C', HPtype.CONTINUOUS, [C]),
                                       'kernel': Hyperparameter('kernel', HPtype.CATEGORICAL, [kernel]),
                                       'degree': Hyperparameter('degree', HPtype.DISCRETE, [degree]),
                                       'gamma': Hyperparameter('gamma', HPtype.CONTINUOUS, [gamma]),
                                       'coef0': Hyperparameter('coef0', HPtype.CONTINUOUS, [coef0])})

        elif kernel == 'sigmoid':
            super(SVM, self).__init__({'C': Hyperparameter('C', HPtype.CONTINUOUS, [C]),
                                       'kernel': Hyperparameter('kernel', HPtype.CATEGORICAL, [kernel]),
                                       'gamma': Hyperparameter('gamma', HPtype.CONTINUOUS, [gamma]),
                                       'coef0': Hyperparameter('coef0', HPtype.CONTINUOUS, [coef0])})

        else:
            raise Exception('No such kernel ("{}") implemented'.format(kernel))

    def fit(self, X_train, t_train):

        """
        Train our model

        :param X_train: NxD numpy array of the observations of the training set {N : nb of obs, D : nb of dimensions}
        :param t_train: Nx1 numpy array of classes associated with each observation in the training set
        """

        self.model_frame.fit(X_train, t_train)

    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array of classes predicted for each observation

        """

        return self.model_frame.predict(X)

    def set_hyperparameters(self, hyperparams):

        """
        Change hyper-parameters of our model

        :param hyperparams: Dictionary of hyper-parameters to change
        """

        self.model_frame.set_params(**hyperparams)



class MLP(Model):

    def __init__(self, hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, momentum=0.9,
                 beta_1=0.9, beta_2=0.999):

        """

        Class that generates a Multi-layer Perceptron classifier.

        This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

        Some hyper-parameters are conditional to others!
        For more information take a look at :
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

        self.model_frame = nn.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                            alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                                            learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
                                            momentum=momentum, beta_1=beta_1, beta_2=beta_2)

        if solver == 'adam':
            super(MLP, self).__init__(
                {'hidden_layer_sizes': Hyperparameter('hidden_layer_sizes', HPtype.CATEGORICAL, [hidden_layer_sizes]),
                 'activation': Hyperparameter('activation', HPtype.CATEGORICAL, [activation]),
                 'solver': Hyperparameter('solver', HPtype.CATEGORICAL, [solver]),
                 'alpha': Hyperparameter('alpha', HPtype.CONTINUOUS, [alpha]),
                 'batch_size': Hyperparameter('batch_size', HPtype.DISCRETE, [batch_size]),
                 'learning_rate_init': Hyperparameter('learning_rate_init', HPtype.CONTINUOUS, [learning_rate_init]),
                 'beta_1': Hyperparameter('beta_1', HPtype.CONTINUOUS, [beta_1]),
                 'beta_2': Hyperparameter('beta_2', HPtype.CONTINUOUS, [beta_2])})

        elif solver == 'sgd':
            super(MLP, self).__init__(
                {'hidden_layer_sizes': Hyperparameter('hidden_layer_sizes', HPtype.CATEGORICAL, [hidden_layer_sizes]),
                 'activation': Hyperparameter('activation', HPtype.CATEGORICAL, [activation]),
                 'solver': Hyperparameter('solver', HPtype.CATEGORICAL, [solver]),
                 'alpha': Hyperparameter('alpha', HPtype.CONTINUOUS, [alpha]),
                 'batch_size': Hyperparameter('batch_size', HPtype.DISCRETE, [batch_size]),
                 'learning_rate': Hyperparameter('learning_rate', HPtype.CATEGORICAL, [learning_rate]),
                 'learning_rate_init': Hyperparameter('learning_rate_init', HPtype.CONTINUOUS, [learning_rate_init]),
                 'power_t': Hyperparameter('power_t', HPtype.CONTINUOUS, [power_t]),
                 'momentum': Hyperparameter('momentum', HPtype.CONTINUOUS, [momentum])})

        elif solver == 'lbfgs':
            super(MLP, self).__init__(
                {'hidden_layer_sizes': Hyperparameter('hidden_layer_sizes', HPtype.CATEGORICAL, [hidden_layer_sizes]),
                 'activation': Hyperparameter('activation', HPtype.CATEGORICAL, [activation]),
                 'solver': Hyperparameter('solver', HPtype.CATEGORICAL, [solver]),
                 'alpha': Hyperparameter('alpha', HPtype.CONTINUOUS, [alpha])})

    def fit(self, X_train, t_train):

        """
        Train our model

        :param X_train: NxD numpy array of the observations of the training set
        :param t_train: Nx1 numpy array classes associated with each observations in the training set
        """

        self.model_frame.fit(X_train, t_train)

    def predict(self, X):

        """
        Predict classes for our observations in the input array X

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :return: Nx1 numpy array classes predicted for each observation

        """

        return self.model_frame.predict(X)

    def set_hyperparameters(self, hyperparams):

        """
        Change hyper-parameters of our model

        :param hyperparams: Dictionary of hyper-parameters to change
        """

        self.model_frame.set_params(**hyperparams)

