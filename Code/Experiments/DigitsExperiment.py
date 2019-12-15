"""
    @file:              DigitsExperiment.py
    @Author:            Nicolas Raymond
    @Creation Date:     12/12/2019
    @Last modification: 12/12/2019

    @Description:       For this experiment, we will evaluate the performance of all hyper-parameter optimization
                        methods implemented in a simple context with a fixed total budget of
                        (250x300x2 = 150 000 epochs), a max budget per config of 600 epochs and a number
                        of 2 cross validation per config tested. For a better understanding the budget will allow
                        our model to evaluate 250 different configurations with non-bandit optimization method.

                        Now, considering Digits data set classification problem (1797 instances, 8x8 images, 10 classes)
                        , will we initialize a Sklearn SVM with default parameter (rbf kernel) and try to find
                        the best values for C (Penalty of the error term) and gamma (kernel coefficient)

                        NOTE : We do not include tuning of best kernel type because not all optimization methods allow
                               to active some parameter search only if the type of kernel associated to them is
                               concerned.
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
x_train, t_train, x_test, t_test = dm.load_digits_dataset()

dataset_name = 'Digits'
nb_cross_validation = 2
experiment_title = 'Digits'
total_budget = 150000
max_budget_per_config = 600

# We initialize an MLP with default hyper-parameters and 4 hidden layers of 20 neurons to classify our data
# and test its performance on both training and test data sets
svm = mod.SVM()


search_space = {'C': ContinuousDomain(-8, 0, log_scaled=True),
                'gamma': ContinuousDomain(-8, 0, log_scaled=True)}

grid_search_space = {'C': DiscreteDomain(list(linspace(10 ** -8, 1, 16))),
                     'gamma': DiscreteDomain(list(linspace(10 ** -8, 1, 16)))}

run_experiment(model=svm, experiment_title=experiment_title, x_train=x_train, t_train=t_train,
               x_test=x_test, t_test=t_test, search_space=search_space, grid_search_space=grid_search_space,
               total_budget=total_budget, max_budget_per_config=max_budget_per_config,
               dataset_name=dataset_name, nb_cross_validation=nb_cross_validation, train_size=len(x_train))
