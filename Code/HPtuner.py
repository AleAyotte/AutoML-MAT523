"""
    @file:              HPtuner.py
    @Author:            Nicolas Raymond
    @Creation Date:     01/10/2019
    @Last modification: 09/10/2019
    @Description:       This file provides all functions linked to hyperparameters optimization methods
"""

import sklearn as sk

class HPtuner:

    def __init__(self, model, method):

        """
        Class that generate an automatic hyperparameter tuner for the model specified

        :param model: Model on which we want to optimize hyperparameters {SVM, MLP} # Work in progress
        :param method: Name of the method of optimization to use {'grid_search', 'random_search'} # Work in progress
        """
        if method not in ['grid_search', 'random_search']:
            raise Exception('No such method "{}" implemented for HPtuner'.format(method))


        self.model = model
        self.method = method


    def set_search_space(self, hp_search_space_dict):

        """

        Function that defines hyperparameter's possible values (or distribution) for all hyperparameters in our model attribute

        :param hp_search_space_dict: Dictionary specifing hyperparameters to tune and search space associate to each of them.
                                     Search space must be a list of values or a statistical distribution from scipy stats.

        :return: Change hyperparameter space in our model attribute

        """

        for hp in hp_search_space_dict:
            self.set_single_hp_space(hp, hp_search_space_dict[hp])


    def set_single_hp_space(self, hyperparameter, space):

        """
        Function that defines hyperparameter's possible values (or distribution) in our model attribute

        :param hyperparameter: Name of the hyperparameter
        :param space: List of values or statistical distribution from scipy.stats

        :return: Change value associate with the hyperparameter in our model attribute HP_space dictionary

        """

        if hyperparameter in self.model.HP_space:
            self.model.HP_space[hyperparameter] = space

        else:
            raise Exception('No such hyperparameter "{}" in our model'.format(hyperparameter))


    def grid_search_sklearn(self, X, t):

        """
        Tune our model with grid search method according to hyperparameters' space

        :param X: NxD numpy array of observations {N : nb of obs; D : nb of dimensions}
        :param X: Nx1 numpy array of target values associated with each observation
        """

        # We find the selection of best hyperparameters according to grid_search
        gs_cv = sk.model_selection.GridSearchCV(self.model.model_frame, self.model.HP_space)
        gs_cv.fit(X,t)

        # We apply changes to original model
        self.model.model_frame = gs_cv.best_estimator_
        self.model.HP_space = gs_cv.best_params_


    def random_search_sklearn(self, X, t, n_iter):

        """
        Tune our model with random search method according to hyperparameters' space

        :param X: NxD numpy array of observations {N : nb of obs; D : nb of dimensions}
        :param X: Nx1 numpy array of target values associated with each observation
        :param n_iter: Number of iterations to do.
        """

        # We find the selection of best hyperparameters according to random_search
        rs_cv = sk.model_selection.RandomizedSearchCV(self.model.model_frame, self.model.HP_space, n_iter)
        rs_cv.fit(X,t)

        # We apply changes to original model
        self.model.model_frame = rs_cv.best_estimator_
        self.model.HP_space = rs_cv.best_params_



    def tune(self, X, t, n_iter = 10):

        """
        Optimize model's hyperparameters with the method specified at the ignition of our tuner

        :param X: NxD numpy array of observations {N : nb of obs; D : nb of dimensions}
        :param X: Nx1 numpy array of target values associated with each observation
        :param n_iter: Number of iteration to do. Only considered if method is 'random_search'
        """

        if self.method == 'grid_search':
            self.grid_search_sklearn(X,t)

        elif self.method == 'random_search':
            self.random_search_sklearn(X,t,n_iter)
