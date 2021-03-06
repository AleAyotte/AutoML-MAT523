"""
    @file:              HPtuner.py
    @Author:            Nicolas Raymond
                        Alexandre Ayotte
    @Creation Date:     01/10/2019
    @Last modification: 02/11/2019
    @Description:       This file provides all functions linked to hyper-parameters optimization methods
"""

import pickle
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from sklearn.model_selection import ParameterGrid
from hyperopt import hp, fmin, rand, tpe, anneal
from Model import HPtype
from enum import Enum, unique
from copy import deepcopy
from tqdm import tqdm
from GPyOpt.methods import BayesianOptimization
from ResultManagement import ExperimentAnalyst
from Worker import start_hpbandster_process

method_list = ['grid_search', 'random_search', 'gaussian_process', 'tpe', 'annealing', 'hyperband', 'BOHB']
domain_type_list = ['ContinuousDomain', 'DiscreteDomain', 'CategoricalDomain']
gaussian_process_methods = ['GP', 'GP_MCMC']
acquistions_type = ['EI', 'MPI']


class HPtuner:

    def __init__(self, model, method, total_budget=10000, max_budget_per_config=200, test_default_hyperparam=False):

        """
        Class that generates an automatic hyper-parameter tuner for the model specified

        :param model: Model on which we want to optimize hyper-parameters
        :param method: Name of the method of optimization to use
        :param total_budget: total budget of our tuner in number of epochs
        :param max_budget_per_config: maximum number of epochs allowed to test a single configuration
        :param test_default_hyperparam: boolean indicating if we should consider default parameter in the tuning
        """

        if method not in method_list:
            raise Exception('No such method "{}" implemented for HPtuner'.format(method))

        self.model = model
        self.method = method
        self.search_space = self.search_space_ignition(method, model)
        self.search_space_modified = False
        self.log_scaled_hyperparameters = False
        self.test_default = test_default_hyperparam
        self.total_budget = total_budget
        self.max_budget_per_config = max_budget_per_config
        self.tuning_history = ExperimentAnalyst(method, type(model).__name__, total_budget, max_budget_per_config)

    def set_search_space(self, hp_search_space_dict):

        """

        Function that defines hyper-parameter's possible values (or distribution) for all hyper-parameters in our model
        attribute

        :param hp_search_space_dict: Dictionary specifing hyper-parameters to tune and domains
                                     associated to each of them. Each domain must be one among
                                     ('ContinuousDomain', 'DiscreteDomain')

        :return: Change hyper-parameter domain in our model attribute
        """

        # We reset search space to default if it has been modified (the state at the ignition)
        if self.search_space_modified:
            self.search_space.reset()
            self.tuning_history.reset()

        # We set every search space one by one
        for hyperparam in hp_search_space_dict:
            self.set_single_hp_space(hyperparam, hp_search_space_dict[hyperparam])

    def set_single_hp_space(self, hyperparameter, domain):

        """
        Function that defines hyper-parameter's possible values (or possible range) in our model attribute

        :param hyperparameter: Name of the hyper-parameter
        :param domain: One domain among ('ContinuousDomain', 'DiscreteDomain')

        :return: Change value associated with the hyper-parameter in our model attribute HP_space dictionary
        """

        # We look if the domain has a compatible format
        if type(domain).__name__ not in domain_type_list:
            raise Exception('No such space type accepted. Must be in {}'.format(domain_type_list))

        # We verifiy if the hyper-parameter exist in our model
        if hyperparameter not in self.model.HP_space:
            raise Exception('No such hyper-parameter "{}" in our model'.format(hyperparameter))

        # We verify if the user want to attribute a continous search space to a discrete or categorical hyper-parameter
        if domain.type.value < self.model.HP_space[hyperparameter].type.value:
            raise Exception('You cannot attribute a continuous search space to a non real hyper-parameter')

        # If the new domain is continuous the current domain will be changed cause
        # the default domain is discrete (NOTE THAT THIS LINE IS ONLY EFFECTIVE WITH GPYOPT SEARCH SPACE)
        if domain.type == DomainType.continuous:
            self.search_space.change_hyperparameter_type(hyperparameter, domain.type)
            if domain.log_scaled:
                self.log_scaled_hyperparameters = True
                self.search_space.save_as_log_scaled(hyperparameter)

        # We change hyper-parameter's domain
        self.search_space[hyperparameter] = domain.compatible_format(self.method, hyperparameter)

    def grid_search(self, loss):

        """
        Tunes our model by testing all possible combination in our search space

        :param loss: loss function to minimize
        """

        # We build all possible configurations
        all_configs = ParameterGrid(self.search_space.space)

        # We find the selection of best hyper-parameters according to grid_search
        print("\n\n")
        pbar = tqdm(total=len(all_configs), postfix='best loss : ' +
                                                    str(round(1 - self.tuning_history.actual_best_accuracy, 10)))

        for config in all_configs:
            loss(config)
            pbar.postfix = 'best loss : ' + str(round(1 - self.tuning_history.actual_best_accuracy, 10))
            pbar.update()

    def random_search(self, loss):

        """
        Tunes our model's hyper-parameters by evaluate random points in our search "n_evals" times

        :param loss: loss function to minimize
        """

        # We find the selection of best hyperparameters according to random_search
        fmin(fn=loss, space=self.search_space.space, algo=rand.suggest, max_evals=self.nb_configs())

    def tpe(self, loss):

        """
        Tunes our model's hyper-parameter with Tree of Parzen estimators method (tpe)

        :param loss: loss function to minimize
        """
        # We find the selection of best hyperparameters according to tpe
        fmin(fn=loss, space=self.search_space.space, algo=tpe.suggest, max_evals=self.nb_configs())

    def simulated_annealing(self, loss):

        """
        Tunes our model's hyper-parameter with simulated annealing derivative free optimization method (SA)

        :param loss: loss function to minimize
        """
        # We find the selection of best hyperparameters according to tpe
        fmin(fn=loss, space=self.search_space.space, algo=anneal.suggest, max_evals=self.nb_configs())

    def gaussian_process(self, loss, **kwargs):

        """
        Tunes our model's hyper-parameter using gaussian process (a bayesian optimization method)

        :param loss: loss function to minimize
        :param kwargs: - nbr_initial_evals : number of points to evaluate before the beginning of the test
                       - method_type : one method available in ['GP', 'GP_MCMC']
                       - acquisition_function : one function available in ['EI', 'MPI']
        """

        # We look for extra parameters
        nbr_initial_evals = kwargs.get('nbr_initial_evals', 5)
        method_type = kwargs.get('method_type', 'GP')
        acquisition_fct = kwargs.get('acquisition_function', 'EI')

        # We verify if values are eligible for the method
        if not isinstance(nbr_initial_evals, int) or nbr_initial_evals < 0:
            raise Exception('Value passed as nbr_initial_evals is not a positive integer')

        if method_type not in gaussian_process_methods:
            raise Exception('Gaussian process method must be in {}'. format(gaussian_process_methods))

        if acquisition_fct not in acquistions_type:
            raise Exception('Acquisition function must be in {}'.format(acquistions_type))

        # We update tuning history method type
        self.tuning_history.method_type = method_type + acquisition_fct

        # We make sure that acquisition function en method type fit together
        if method_type == 'GP_MCMC':
            acquisition_fct += '_MCMC'

        # We execute the hyper-parameter optimization
        optimizer = BayesianOptimization(loss, domain=self.search_space.space,
                                         model_type=method_type, initial_design_numdata=nbr_initial_evals,
                                         acquisition_type=acquisition_fct)

        optimizer.run_optimization(max_iter=(self.nb_configs() - nbr_initial_evals))
        optimizer.plot_acquisition()

    def hyperband(self, loss):

        """
        Tune's our model's hyper-parameters using hyperband method

        :param loss: loss function to minimize
        :param n_evals: maximal number of evaluations to do (budget)
        """
        # We initialize the process
        NS, smax, optimizer = start_hpbandster_process(self.method, self.search_space.space, loss,
                                                       self.total_budget, self.max_budget_per_config)
        # We run the optimization
        res = optimizer.run(n_iterations=smax)

        # We extract the results
        id2config = res.get_id2config_mapping()

        # We update total unique sampled number of our Tuning History (Experiment Analyst object)
        self.tuning_history.total_unique_sampled = len(id2config.keys())
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        optimizer.shutdown(shutdown_workers=True)
        NS.shutdown()

    def tune(self, X=None, t=None, dtset=None, nb_cross_validation=1, valid_size=0.2, **kwargs):

        """
        Optimizes model's hyper-parameters with the method specified at the ignition of our tuner

        :param X: NxD numpy array of observations {N : nb of obs; D : nb of dimensions}
        :param t: Nx1 numpy array of target values associated with each observation
        :param dtset: A torch dataset which contain our train data points and labels
        :param nb_cross_validation: Number of cross validation done for loss calculation
        :param valid_size: Percentage of training data used as validation data
        """

        # We change our model maximal number of epochs to do in training
        self.model.set_max_epoch(int(self.max_budget_per_config / nb_cross_validation))

        # We set the number of cross validation and valid size used, in tuning history
        self.tuning_history.nbr_of_cross_validation = nb_cross_validation
        self.tuning_history.validation_size = valid_size

        # We save results for the default hyperparameters if the user wanted it
        if self.test_default:
            self.total_budget -= self.max_budget_per_config
            self.test_default_hyperparameters(X, t, dtset, nb_cross_validation, valid_size)

        # We reformat the search space
        self.search_space.reformat_for_tuning()

        # We build loss function
        loss = self.build_loss_funct(X=X, t=t, dtset=dtset,
                                     nb_of_cross_validation=nb_cross_validation, valid_size=valid_size)

        # We tune hyper-parameters with the method chosen
        if self.method == 'grid_search':
            self.grid_search(loss)

        elif self.method == 'random_search':
            self.random_search(loss)

        elif self.method == 'tpe':
            self.tpe(loss)

        elif self.method == 'gaussian_process':
            self.gaussian_process(loss, **kwargs)

        elif self.method == 'annealing':
            self.simulated_annealing(loss)

        elif self.method == 'hyperband' or self.method == 'BOHB':
            self.hyperband(loss)

        else:
            raise NotImplementedError

        # We save best hyper-parameters
        best_hyperparameters = self.tuning_history.best_hyperparameters

        # We apply changes to original model
        self.model.set_hyperparameters(best_hyperparameters)

        # We train the model a last time with the best hyper-parameters
        print("\n\n")
        print("Training with best hyper-parameters found in processing...")
        self.model.fit(X, t, dtset)

        return self.tuning_history

    @staticmethod
    def search_space_ignition(method, model):

        """
        Defines a correct search space format according to optimization method.

        :return: Search space frame for our tuner
        """

        if method == 'grid_search':

            return SklearnSearchSpace(model)

        elif method in ['random_search', 'tpe', 'annealing']:

            return HyperoptSearchSpace(model)

        elif method == 'gaussian_process' or method == 'random_forest':

            return GPyOptSearchSpace(model)

        elif method == 'hyperband' or method == 'BOHB':

            return HpBandSterSearchSpace(model)

        else:
            raise NotImplementedError

    def build_loss_funct(self, X=None, t=None, dtset=None, nb_of_cross_validation=1, valid_size=0.2):

        """
        Builds a loss function, returning the mean of a cross validation, that will be available for HPtuner methods

        :param X: NxD numpy array of observations {N : nb of obs, D : nb of dimensions}
        :param t: Nx1 numpy array of classes associated with each observation
        :param nb_of_cross_validation: Number of data splits and validation to execute
        :param dtset: A torch dataset which contain our train data points and labels
        :param valid_size: percentage of training data used as validation data
        :return: A specific loss function for our tuner
        """

        if self.method in ['grid_search', 'random_search', 'tpe', 'annealing']:

            def loss(hyperparams):
                """
                Returns the mean negative value of the accuracy on a cross validation
                (minimize (1 - accuracy) is equivalent to maximize accuracy)

                :param hyperparams: dict of hyper-parameters
                :return: 1 - (mean accuracy on cross validation)
                """

                if self.log_scaled_hyperparameters:
                    self.exponential(hyperparams, self.search_space.log_scaled_hyperparam)

                # We set the hyper-parameters and compute the loss associated to it
                self.model.set_hyperparameters(hyperparams)
                loss_value = 1 - (self.model.cross_validation(X_train=X, t_train=t, dtset=dtset,
                                                              nb_of_cross_validation=nb_of_cross_validation,
                                                              valid_size=valid_size))
                # We update our tuning history
                self.tuning_history.update(loss_value, hyperparams, self.max_budget_per_config)

                return loss_value

            return loss

        elif self.method == 'gaussian_process':

            # We start a loading bar
            print("\n\n")
            pbar = tqdm(total=self.nb_configs(), postfix='best loss : ' +
                        str(round(1 - self.tuning_history.actual_best_accuracy, 10)))

            def loss(hyperparams):

                """
                Returns the mean negative value of the accuracy on a cross validation
                (minimize 1 - accuracy is equivalent to maximize accuracy)

                :param hyperparams: 2d-numpy array containing only values of hyper-parameters
                :return: 1 - (mean accuracy on cross validation)
                """
                # We extract the values from the 2d-numpy array
                hyperparams = self.search_space.change_to_dict(hyperparams)

                if self.log_scaled_hyperparameters:
                    self.exponential(hyperparams, self.search_space.log_scaled_hyperparam)

                # If some integer hyper-parameter are considered as numpy.float64 we convert them as int
                self.float_to_int(hyperparams)

                # We set the hyper-parameters and compute the loss associated to it
                self.model.set_hyperparameters(hyperparams)
                loss_value = 1 - (self.model.cross_validation(X_train=X, t_train=t, dtset=dtset,
                                                              nb_of_cross_validation=nb_of_cross_validation,
                                                              valid_size=valid_size))

                # We update our tuning history and the loading bar
                self.tuning_history.update(loss_value, hyperparams, self.max_budget_per_config)
                pbar.postfix = 'best loss : ' + str(round(1 - self.tuning_history.actual_best_accuracy, 10))
                pbar.update()

                return loss_value

            return loss

        elif self.method == 'hyperband' or self.method == 'BOHB':

            pickle_obj = pickle.dumps(self.model)

            def loss(hyperparams, budget):

                """
                Returns the mean negative value of the accuracy on a cross validation
                (minimize 1 - accuracy is equivalent to maximize accuracy)

                :param hyperparams: dict of hyper-parameters
                :param budget: maximal number of epoch allowed for the model training
                :return: 1 - (mean accuracy on cross validation)

                """
                # We start a new model base on default parameter of our original model
                copied_model = pickle.loads(pickle_obj)

                if self.log_scaled_hyperparameters:
                    self.exponential(hyperparams, self.search_space.log_scaled_hyperparam)

                # If some integer hyper-parameter are considered as numpy.float64 we convert them as int
                self.float_to_int(hyperparams)

                # We add or change the parameter "max_iter"
                copied_model.set_max_epoch(int(budget))

                # We set the hyper-parameters and compute the loss associated to it
                copied_model.set_hyperparameters(hyperparams)
                loss_value = 1 - (copied_model.cross_validation(X_train=X, t_train=t, dtset=dtset,
                                                                nb_of_cross_validation=nb_of_cross_validation,
                                                                valid_size=valid_size))

                self.tuning_history.update(loss_value, hyperparams, int(budget))

                return loss_value

            return loss

        else:
            raise NotImplementedError

    def test_default_hyperparameters(self, X, t, dtset, nb_cross_validation, valid_size):

        """
        Calculates loss according to default hyper-parameters

        :param nb_cross_validation: Number of data splits and validation to execute
        """

        default_loss_value = self.model.cross_validation(X, t, dtset, valid_size=valid_size,
                                                         nb_of_cross_validation=nb_cross_validation)
        search_space = SklearnSearchSpace(self.model)
        self.tuning_history.update(default_loss_value, search_space.space)

    @staticmethod
    def exponential(original_hp_dict, list_of_log_scaled_hp):

        """
        Transforms log_scaled hyper-parameter as a power of 10

        :param original_hp_dict: hyper-parameter dictionary
        :param list_of_log_scaled_hp: list of hyper-parameters's name to transform
        :return:
        """
        for hyperparam in list_of_log_scaled_hp:
            original_hp_dict[hyperparam] = 10 ** original_hp_dict[hyperparam]

    def float_to_int(self, hp_dict):

        """
        If an integer hyper-paramater is considered as a float, it will be converted as int.
        Solves float problem caused by GaussianProcess algorithm.

        :param hp_dict: hyper-parameter dictionary to fix
        """
        for hyperparam in hp_dict:
            if self.model.HP_space[hyperparam].type.value == HPtype.integer.value:
                hp_dict[hyperparam] = int(hp_dict[hyperparam])

    def nb_configs(self):

        """
        Compute the equivalence of the epochs budget in terms of number points to evaluate in our search space
        for all method except hyperband and bohb

        :return: number of point to evaluate at the maximal budget per config
        """
        return int(max(self.total_budget/self.max_budget_per_config, 1))


class SearchSpace:

    def __init__(self, space):

        """
        Definition of a search space for our hyper-parameters
        """

        self.default_space = space
        self.space = space
        self.log_scaled_hyperparam = []

    def reset(self):

        """
        Resets search space to default
        """

        self.space = deepcopy(self.default_space)
        self.log_scaled_hyperparam.clear()

    def change_hyperparameter_type(self, hyperparam, new_type):

        """
        Changes hyper-parameter type in search space (only useful in GPyOpt search spaces)

        :param hyperparam: Name of the hyperparameter
        :param new_type: Type from HPtype
        """

        pass

    def reformat_for_tuning(self):

        """
        Reformats search space so it is now compatible with hyper-parameter optimization method
        """

        pass

    def save_as_log_scaled(self, hyperparam):

        """
        Saves hyper-parameter's name that is log scaled

        :param hyperparam: Name of the hyperparameter
        """

        self.log_scaled_hyperparam.append(hyperparam)

    def __getitem__(self, key):
        return self.space[key]

    def __setitem__(self, key, value):
        self.space[key] = value


class HyperoptSearchSpace(SearchSpace):

    def __init__(self, model):

        """
        Class that defines a compatible search space with Hyperopt package hyper-parameter optimization algorithm

        :param model: Available model from Model.py
        """

        space = {}

        for hyperparam in model.HP_space:
            space[hyperparam] = model.HP_space[hyperparam]

        super(HyperoptSearchSpace, self).__init__(space)

    def reformat_for_tuning(self):

        """
        Inserts the whole built space in a hp.choice object that can now be pass as a space parameter
        in Hyperopt hyper-parameter optimization algorithm
        """

        for hyperparam in list(self.space.keys()):

            # We check if the hyper-parameter space is an hyperOpt object (if yes, the user wants it to be tune)
            if type(self[hyperparam]).__name__ == 'Hyperparameter':
                self.space.pop(hyperparam)

        self.space = hp.choice('space', [self.space])


class SklearnSearchSpace(SearchSpace):

    def __init__(self, model):

        """
        Class that defines a compatible search space with Sklearn package hyper-parameter optimization algorithm

        :param model: Available model from Model.py
        """

        space = {}

        for hyperparam in model.HP_space:
            space[hyperparam] = model.HP_space[hyperparam].value

        super(SklearnSearchSpace, self).__init__(space)


class GPyOptSearchSpace(SearchSpace):

    def __init__(self, model):

        """
        Class that defines a compatible search space with GPyOpt package hyper-parameter optimization algorithm

        :param model: Available model from Model.py
        """

        space = {}
        self.categorical_vars = {}

        for hyperparam in model.HP_space:

            hp_initial_value = model.HP_space[hyperparam].value[0]

            if model.HP_space[hyperparam].type.value == HPtype.categorical.value:

                space[hyperparam] = {'name': hyperparam, 'type': 'categorical',
                                     'domain': (hp_initial_value,), 'dimensionality': 1}

                self.categorical_vars[hyperparam] = {}

            else:
                space[hyperparam] = {'name': hyperparam, 'type': 'discrete',
                                     'domain': (hp_initial_value,), 'dimensionality': 1}

        super(GPyOptSearchSpace, self).__init__(space)
        self.hyperparameters_to_tune = None

    def change_hyperparameter_type(self, hp_to_fix, new_type):

        """
        Changes hyper-parameter type in the search space

        :param hp_to_fix: Name of the hyper-parameter which we want to change his type
        :param new_type: The new type (one among DomainType)
        """
        self[hp_to_fix]['type'] = new_type.name

    def reformat_for_tuning(self):

        """
        Converts the dictionnary to a list containing only internal dictionaries.
        Only keep hyper-parameters that has more than a unique discrete value as a domain
        """

        for hyperparam in list(self.space.keys()):

            # We save the length of the domain (must be at least 2)
            domain_length = len(self[hyperparam]['domain'])

            # If there's no search to be done with the hyper-parameter we do not consider it anymore in the tuning
            if domain_length == 1:
                self.space.pop(hyperparam)

            # If the hyper-parameter is categorical, we change strings for integer.
            elif self[hyperparam]['type'] == 'categorical':

                # We save the possible values of the categorical variables in forms of strings and also integers
                choices = list(self[hyperparam]['domain'])
                integer_encoding = tuple(range(domain_length))

                # We change the domain of our space for the tuple with all values (int) possible
                self[hyperparam]['domain'] = integer_encoding

                # We save the choices associated with each integer in our dictionary
                for i in integer_encoding:
                    self.categorical_vars[hyperparam][i] = choices[i]

        self.hyperparameters_to_tune = list(self.space.keys())
        self.space = list(self.space.values())
        print(self.hyperparameters_to_tune)

        if len(self.hyperparameters_to_tune) == 0:
            raise Exception('The search space has not been modified yet. Each hyper-parameter has only a discrete'
                            'domain of length 1 and no tuning can be done yet')

    def change_to_dict(self, hyper_paramater_values):

        """
        Builds a dictionary of hyper-parameters
        :param hyper_paramater_values: 2d numpy array of hyper-parameters' values
        :return: dictionary of hyper-parameters
        """

        # We initialize a dictionary and an index
        hp_dict, i = {}, 0

        # We extract hyper-parameters' values
        hyper_parameter_values = hyper_paramater_values[0]

        # We build the dict and transform back categorical variables
        for hyperparam in self.hyperparameters_to_tune:

            hp_value = hyper_parameter_values[i]  # Represents a dictionary key for categorical vars

            if hyperparam in self.categorical_vars:
                hp_dict[hyperparam] = self.categorical_vars[hyperparam][hp_value]

            else:
                hp_dict[hyperparam] = hp_value

            i += 1

        return hp_dict

    def __setitem__(self, key, value):
        self.space[key]['domain'] = value


class HpBandSterSearchSpace(SearchSpace):

    def __init__(self, model):

        """
        Class that defines a compatible search space with HpBandSter package hyper-parameter optimization algorithm

        :param model: Available model from Model.py
        """

        space = {}

        super(HpBandSterSearchSpace, self).__init__(space)

    def reformat_for_tuning(self):

        """
        Converts the dictionnary of CSH object to a proper ConfigurationSpace accepted by HpBandSter.
        """

        # Initialization of configuration space
        cs = CS.ConfigurationSpace()

        # We extract CSH object from the dictionnary and put it in a list
        if len(self.space) != 0:
            self.space = list(self.space.values())
            cs.add_hyperparameters(self.space)
            self.space = cs

        else:
            raise Exception('Search space has not been modified yet, no tuning can be done.')


@unique
class DomainType(Enum):

    """
    Class containing possible types of hyper-parameters
    """

    continuous = 1
    discrete = 3


class Domain:

    def __init__(self, type):

        """
        Abstract (parent) class that represents a domain for hyper-parameter's possible values

        :param type: One type of domain among DomainType
        """

        self.type = type


class ContinuousDomain(Domain):

    def __init__(self, lower_bound, upper_bound, log_scaled=False):

        """
        Class that generates a continuous domain

        :param lower_bound: Lowest possible value (included)
        :param upper_bound: Highest possible value (included)
        :param log_scaled: If True, hyper-parameter will now be seen as 10^x where x follows a uniform(lb,ub)
        """

        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        self.lb = lower_bound
        self.ub = upper_bound
        self.log_scaled = log_scaled

        super(ContinuousDomain, self).__init__(DomainType.continuous)

    def compatible_format(self, tuner_method, label):

        """
        Builds the correct format of a uniform distribution according to the method used by the tuner

        :param tuner_method: Name of the method employed by the HPtuner.
        :param label: String defining the name of the hyper-parameter
        :return: Uniform distribution compatible with method used by HPtuner
        """

        if tuner_method in ['random_search', 'tpe', 'annealing']:
            return hp.uniform(label, self.lb, self.ub)

        elif tuner_method == 'gaussian_process' or tuner_method == 'random_forest':
            return tuple([self.lb, self.ub])

        elif tuner_method == 'hyperband' or tuner_method == 'BOHB':
            return CSH.UniformFloatHyperparameter(label, lower=self.lb, upper=self.ub, log=False)


class DiscreteDomain(Domain):

    def __init__(self, possible_values):

        """
        Class that generates a domain with possible discrete values of an hyper-parameter

        :param possible_values: list of values


        """
        self.values = possible_values

        super(DiscreteDomain, self).__init__(DomainType.discrete)

    def compatible_format(self, tuner_method, label):

        """
        Builds the correct format of discrete set of values according to the method used by the tuner

        :param tuner_method: Name of the method employed by the HPtuner.
        :param label: String defining the name of the hyper-parameter
        :return: Set of values compatible with method used by HPtuner
        """

        if tuner_method == 'grid_search':
            return self.values

        elif tuner_method in ['random_search', 'tpe', 'annealing']:
            return hp.choice(label, self.values)

        elif tuner_method == 'gaussian_process' or tuner_method == 'random_forest':
            return tuple(self.values)

        elif tuner_method == 'hyperband' or tuner_method == 'BOHB':
            return CSH.CategoricalHyperparameter(label, choices=self.values)
