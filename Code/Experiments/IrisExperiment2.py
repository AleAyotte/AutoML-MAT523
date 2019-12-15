"""
    @file:              IrisExperiment.py
    @Author:            Nicolas Raymond
    @Creation Date:     02/12/2019
    @Last modification: 12/12/2019

    @Description:       For this experiment, we will evaluate the performance of all hyper-parameter optimization
                        methods implemented in a simple context with a fixed total budget of
                        (250x150x4 = 150 000 epochs), a max budget per config of 600 epochs and a number
                        of 4 cross validation per config tested. For a better understanding the budget will allow
                        our model to evaluate 250 different configurations with non-bandit optimization method.

                        Now, considering Iris data set classification problem (150 instances, 4 attributes, 3 classes),
                        will we initialize a Sklearn MLP with 4 hidden layers of 20 neurons with default parameter and
                        try to find the best values for alpha (L2 penalty (regularization term) parameter),
                        learning rate init (initial learning rate used), beta1  (exponential decay rate for estimates of
                        first moment vector in adam), beta2 (exponential decay rate for estimates
                        of second moment vector in adam), hidden layers number (between 1 and 20) and finally layers
                        size (number of neurons on each hidden layer between 5 and 50) with all methods available.

                        NOTE : Not all optimization methods allow us to manage each layer size separately. Because of
                               this we will set the same number of neurons for every layers in our experiment.
"""

# Import code needed
import sys
import os
from numpy import linspace


# Append path of module to sys and import module
sys.path.append(os.getcwd())
module_path = os.path.dirname(os.getcwd())
sys.path.append(module_path)
import DataManager as dm
import Model as mod
from Experiment_frame import run_experiment
from HPtuner import ContinuousDomain, DiscreteDomain


# We generate data for our tests and global variables for all tests
x_train, t_train, x_test, t_test = dm.load_iris_dataset(random_state=42)
dataset_name = 'Iris'
nb_cross_validation = 4
experiment_title = 'IrisClassification2'
total_budget = 150000
max_budget_per_config = 600

# We initialize an MLP with default hyper-parameters and 4 hidden layers of 20 neurons to classify our data
# and test its performance on both training and test data sets
mlp = mod.MLP(hidden_layers_number=4, layers_size=20, max_iter=1000)


search_space = {'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int).tolist())),
                'hidden_layers_number': DiscreteDomain(range(1, 21)),
                'layers_size': DiscreteDomain(range(5, 51))}

grid_search_space = {'alpha': DiscreteDomain(list(linspace(10 ** -8, 1, 5))),
                     'learning_rate_init': DiscreteDomain(list(linspace(10 ** -8, 1, 5))),
                     'batch_size': DiscreteDomain([200]),
                     'hidden_layers_number': DiscreteDomain([1, 5, 10, 15, 20]),
                     'layers_size': DiscreteDomain([20, 50])}


run_experiment(model=mlp, experiment_title=experiment_title, x_train=x_train, t_train=t_train,
               x_test=x_test, t_test=t_test, search_space=search_space, grid_search_space=grid_search_space,
               total_budget=total_budget, max_budget_per_config=max_budget_per_config,
               dataset_name=dataset_name, nb_cross_validation=nb_cross_validation, train_size=len(x_train))
