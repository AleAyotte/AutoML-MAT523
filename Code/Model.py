"""
    @file:              DataGen.py
    @Author:            Nicolas Raymond
    @Creation Date:     30/09/2019
    @Last modification: 30/09/2019
    @Description:       This program generate models of different types to use for our classification problems.

"""

import numpy as np
import sklearn.svm as svm
import sklearn.neural_network as nn
import matplotlib.pyplot as plt

class Model:

    def __init__(self, HP_Dict):

        """
        Class that generate a model to solve a classification problem

        :param HP_Dict: Dictionary containing all hyperparameters of the model

        """
        self.HP_space = HP_Dict


    def train(self, X_train, t_train):

        """
        Train our model

        Note that it wil be override by the children classes

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

    def accuracy(self, X, t):

        """
        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :return: Good classification rate

        """
        predictions = self.predict(X)

        diff = t - predictions

        return ((diff == 0).sum())/len(diff) # (Nb of good predictions / nb of predictions)

    def plot_data(self, data, classes):

        """

        Plot data points and spaces of separation done by the model for 2D cases only.

        :param data: Nx2 numpy array of observations {N : nb of obs}
        :param classes: Classes associate with each data point.
        """

        if data.shape[1] != 2:
            raise Exception('Method only available for 2D plotting (two dimensions datasets')

        else :
            ix = np.arange(data[:, 0].min(), data[:, 0].max(), 0.01)
            iy = np.arange(data[:, 1].min(), data[:, 1].max(), 0.01)
            iX, iY = np.meshgrid(ix, iy)
            x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
            contour_out = self.predict(x_vis)
            contour_out = contour_out.reshape(iX.shape)

            plt.contourf(iX, iY, contour_out)
            plt.scatter(data[:, 0], data[:, 1], s=105, c=classes, edgecolors='b')
            plt.title('Accuracy on test : {} %'.format(self.accuracy(data,classes)*100))
            plt.show()



class SVM(Model):

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma='auto_deprecated', coef0=0.0, max_iter=-1):

        """
        Class that generates a support vector machine

        Some hyperparameters are conditional to others!
        Take a look at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC for
        more informations on hyperparameters

        :param C: Penalty parameter C of the error term
        :param kernel: Kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’ or ‘sigmoid’
        :param degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels
        :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
        :param coef0: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        """

        self.model_frame = svm.SVC(C, kernel, degree, gamma, coef0, max_iter=max_iter)

        if kernel == 'rbf':
            super().__init__({'C':C, 'kernel':kernel, 'gamma':gamma})

        elif kernel == 'linear':
            super().__init__({'C':C, 'kernel':kernel})

        elif kernel == 'poly':
            super().__init__({'C': C, 'kernel': kernel, 'degree':degree, 'gamma': gamma, 'coef0':coef0})

        elif kernel == 'sigmoid':
            super().__init__({'C': C, 'kernel': kernel, 'gamma': gamma, 'coef0': coef0})

        else:
            raise Exception('No such kernel ("{}") implemented'.format(kernel))


    def train(self, X_train, t_train):

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

        self.model_frame = nn.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                                alpha=alpha, batch_size=batch_size,learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                                                power_t=power_t, max_iter=max_iter, momentum=momentum, beta_1=beta_1, beta_2=beta_2)


        if solver == 'adam':
            super().__init__(
                {'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation, 'solver': solver, 'alpha': alpha,
                 'batch_size': batch_size, 'learning_rare_init':learning_rate_init, 'beta_1': beta_1, 'beta_2': beta_2})

        elif solver == 'sgd':
            super().__init__({'hidden_layer_sizes':hidden_layer_sizes, 'activation':activation, 'solver':solver, 'alpha':alpha,
                              'batch_size':batch_size, 'learning_rate':learning_rate, 'learning_rate_init':learning_rate_init,
                              'power_t':power_t,'momentum':momentum})

        elif solver == 'lbfgs':
            super().__init__({'hidden_layer_sizes': hidden_layer_sizes, 'activation': activation, 'solver': solver, 'alpha': alpha})

    def train(self, X_train, t_train):

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