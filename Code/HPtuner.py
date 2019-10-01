"""
    @file:              HPtuner.py
    @Author:            Nicolas Raymond
    @Creation Date:     01/10/2019
    @Last modification: 01/10/2019
    @Description:       This file provide all functions linked to hyperparameters optimization methods
"""

import sklearn as sk

class HPtuner:

    def __init__(self, model, method):

        """
        Class that generate an automatic hyperparameter tuner for the model specified

        :param model: Model on which we want to optimize hyperparameters {SVM, MLP} # Work in progress
        :param method: Method of optimization to use {'grid_search', 'random_search'} # Work in progress
        """

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


    def grid_search_sklearn(self):

        """
        Build a GridSearchCV sklearn object based on our model and his hyperparameters' space

        :return: Change our model for a GridSearchCV
        """

        self.model = sk.model_selection.GridSearchCV(self.model, self.model.HP_space)

    

    def random_search_sklearn(self, n_iter):

        """
        Build a RandomizedSearchCV sklearn object based on our model and his hyperparameters' space

        :return: Change our model for a RandomizedSearchCV

        """

        self.model = sk.model_selection.RandomizedSearchCV(self.model, self.mode.HP_space, n_iter)

