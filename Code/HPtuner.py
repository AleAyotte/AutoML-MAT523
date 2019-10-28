"""
    @file:              HPtuner.py
    @Author:            Nicolas Raymond
    @Creation Date:     01/10/2019
    @Last modification: 09/10/2019
    @Description:       This file provides all functions linked to hyper-parameters optimization methods
"""

import sklearn as sk
from hyperopt import hp, fmin, rand, tpe, space_eval
from Code.Model import HPtype


method_list = ['grid_search', 'random_search', 'gaussian_process', 'tpe', 'random_forest', 'hyperband', 'bohb']
domain_type_list = ['ContinuousDomain', 'DiscreteDomain', 'CategoricalDomain']


class HPtuner:

    def __init__(self, model, method):

        """
        Class that generate an automatic hyper-parameter tuner for the model specified

        :param model: Model on which we want to optimize hyper-parameters {SVM, MLP} # Work in progress
        :param method: Name of the method of optimization to use {'grid_search', 'random_search'} # Work in progress
        """
        if method not in method_list:
            raise Exception('No such method "{}" implemented for HPtuner'.format(method))

        self.model = model
        self.method = method
        self.search_space = self.search_space_ignition(method, model)

    def set_search_space(self, hp_search_space_dict):

        """

        Function that defines hyper-parameter's possible values (or distribution) for all hyper-parameters in our model
        attribute

        :param hp_search_space_dict: Dictionary specifing hyper-parameters to tune and search space
                                     associate to each of them. Search space must be a list of values
                                     or a statistical distribution from scipy stats.

        :return: Change hyper-parameter space in our model attribute

        """

        for hyperparam in hp_search_space_dict:
            self.set_single_hp_space(hyperparam, hp_search_space_dict[hyperparam])

    def set_single_hp_space(self, hyperparameter, space):

        """
        Function that defines hyper-parameter's possible values (or distribution) in our model attribute

        :param hyperparameter: Name of the hyper-parameter
        :param space: List of values or statistical distribution from scipy.stats

        :return: Change value associate with the hyper-parameter in our model attribute HP_space dictionary

        """

        if type(space).__name__ not in domain_type_list:
            raise Exception('No such space type accepted. Must be in {}'.format(domain_type_list))

        if hyperparameter not in self.model.HP_space:
            raise Exception('No such hyper-parameter "{}" in our model'.format(hyperparameter))

        if self.model.HP_space[hyperparameter].type != space.type:
            raise Exception('Conflict between {} type and space type'.format(hyperparameter))

        else:
            self.search_space[hyperparameter] = space.compatible_format(self.method, hyperparameter)

    def grid_search_sklearn(self, X, t):

        """
        Tune our model with grid search method according to hyper-parameters' space

        :param X: NxD numpy array of observations {N : nb of obs; D : nb of dimensions}
        :param t: Nx1 numpy array of target values associated with each observation
        """

        # We find the selection of best hyperparameters according to grid_search
        gs_cv = sk.model_selection.GridSearchCV(self.model.model_frame, self.model.HP_space)
        gs_cv.fit(X, t)

        # We apply changes to original model
        self.model.model_frame = gs_cv.best_estimator_
        self.model.HP_space = gs_cv.best_params_

    def random_search(self, loss, n_evals):

        """
        Tune our model with random search method according to hyperparameters' space

        :param loss: loss function to minimize
        :param n_evals: Number of evaluations to do
        """

        # We find the selection of best hyperparameters according to random_search
        best_hyperparams = fmin(fn=loss, space=self.search_space, algo=rand.suggest, max_evals=n_evals)
        best_hyperparams = space_eval(self.search_space, best_hyperparams)

        # We apply changes to original model
        self.model.set_hyperparameters(best_hyperparams)

    def tune(self, X, t, n_evals=10):

        """
        Optimize model's hyperparameters with the method specified at the ignition of our tuner

        :param X: NxD numpy array of observations {N : nb of obs; D : nb of dimensions}
        :param t: Nx1 numpy array of target values associated with each observation
        :param n_evals: Number of evaluations to do. Only considered if method is 'random_search'
        """

        if self.method == 'grid_search':
            self.grid_search_sklearn(X, t)

        else:

            loss = self.build_loss_funct(X, t)

            if self.method == 'random_search':
                self.random_search(loss, n_evals)

    @staticmethod
    def search_space_ignition(method, model):

        """
        Define a correct search space format according to optimization method

        :return: Search space frame for our tuner

        """

        if method == 'grid_search':

            return SklearnSearchSpace(model)

        elif method == 'random_search' or method == 'tpe':

            return HyperoptSearchSpace(model)

        elif method == 'gaussian_process' or method == 'random_forest':

            return GPyOptSearchSpace(model)

        else:
            raise NotImplementedError

    def build_loss_funct(self, X, t, nb_of_cross_validation=3):

        """
        Build a loss function, returning the mean of a cross validation, that will be available for HPtuner methods

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :param nb_of_cross_validation: Number of data splits and validation to execute
        :return: A specific loss function for our tuner
        """
        if self.method == 'random_search' or self.method == 'tpe':

            def loss(hyperparams):
                self.model.set_hyperparameters(hyperparams)
                return -1*(self.model.cross_validation(X, t, nb_of_cross_validation))

            return loss

        else:
            raise NotImplementedError


class HyperoptSearchSpace:

    def __init__(self, model):

        """
        Class that defines a compatible search space with Hyperopt package hyper-parameter optimization algorithm
        :param model: Available model from Model.py

        """

        hp_dict = {}

        for hyperparam in model.HP_space:
            hp_dict[hyperparam] = hp.choice(hyperparam, model.HP_space[hyperparam].value)

        self.space = hp.choice('space', [hp_dict])

    def __getitem__(self, key):
        return self.space['space'][key]


class SklearnSearchSpace:

    def __init__(self, model):

        """
        Class that defines a compatible search space with Sklearn package hyper-parameter optimization algorithm
        :param model: Available model from Model.py

        """

        space = {}

        for hyperparam in model.HP_space:
            space[hyperparam] = model.HP_space[hyperparam].value

        self.space = space

    def __getitem__(self, key):
        return self.space[key]


class GPyOptSearchSpace:

    def __init__(self, model):

        """
        Class that defines a compatible search space with GPyOpt package hyper-parameter optimization algorithm
        :param model: Available model from Model.py

        """

        space = []

        for hyperparam in model.HP_space:
            hp_initial_value = model.HP_space[hyperparam].value[0]
            hp_type = model.HP_space[hyperparam].type_name

            space.append({'name': hyperparam, 'type': hp_type,
                          'domain': (hp_initial_value,)})

        self.space = space

    def __getitem__(self, key):
        return next(hyperparam for hyperparam in self.space if hyperparam['name'] == key)


class Domain:

    def __init__(self, type):

        """
        Abstract (parent) class that represents a domain for hyper-parameter's possible values

        :param type: One type of domain among HPtype (CONTINUOUS, DISCRETE, CATEGORICAL)
        """
        self.type = type


class ContinuousDomain(Domain):

    def __init__(self, lower_bound, upper_bound):

        """
        Class that generates a continuous domain

        :param lower_bound: Lowest possible value (included)
        :param upper_bound: Highest possible value (included)

        """

        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        self.lb = lower_bound
        self.ub = upper_bound

        super(ContinuousDomain, self).__init__(HPtype.CONTINUOUS)

    def compatible_format(self, tuner_method, label):

        """
        Build the correct format of a uniform distribution according to the method used by the tuner

        :param tuner_method: Name of the method employed by the HPtuner.
        :param label: String defining the name of the hyper-parameter
        :return: Uniform distribution compatible with method used by HPtuner
        """

        if tuner_method == 'random_search' or tuner_method == 'tpe':
            return hp.uniform(label, self.lb, self.ub)

        elif tuner_method == 'gaussian_process' or tuner_method == 'random_forest':
            return tuple([self.lb, self.ub])


class DiscreteDomain(Domain):

    def __init__(self, possible_values):

        """
        Class that generates a domain with possible discrete values of an hyper-parameter
        :param possible_values: list of values

        """

        self.values = possible_values

        super(DiscreteDomain, self).__init__(HPtype.DISCRETE)

    def compatible_format(self, tuner_method, label):

        """
        Build the correct format of discrete set of values according to the method used by the tuner

        :param tuner_method: Name of the method employed by the HPtuner.
        :param label: String defining the name of the hyper-parameter
        :return: Set of values compatible with method used by HPtuner
        """

        if tuner_method == 'random_search' or tuner_method == 'tpe':
            return hp.choice(label, self.values)

        elif tuner_method == 'gaussian_process' or tuner_method == 'random_forest':
            return tuple(self.values)


class CategoricalDomain(Domain):

    def __init__(self, possible_values):

        """
        Class that generates a domain with possible categorical values of an hyper-parameter
        :param possible_values: list of values

        """
        self.values = possible_values

        super(CategoricalDomain, self).__init__(HPtype.CATEGORICAL)

    def compatible_format(self, tuner_method, label):

        """
        Build the correct format of categorical set of values according to the method used by the tuner

        :param tuner_method: Name of the method employed by the HPtuner.
        :param label: String defining the name of the hyper-parameter
        :return: Set of values compatible with method used by HPtuner
        """

        if tuner_method == 'random_search' or tuner_method == 'tpe':
            return hp.choice(label, self.values)

        elif tuner_method == 'gaussian_process' or tuner_method == 'random_forest':
            return tuple(self.values)
