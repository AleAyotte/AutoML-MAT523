"""
    @file:              IrisExperiment.py
    @Author:            Nicolas Raymond
    @Creation Date:     02/12/2019
    @Last modification: 11/12/2019

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

# Append path of module to sys and import module
sys.path.append(os.getcwd())
module_path = os.path.dirname(os.getcwd())
sys.path.append(module_path)
import DataManager as dm
from MLP_experiment_frame import mlp_experiment


# We generate data for our tests and global variables for all tests
x_train, t_train, x_test, t_test = dm.load_iris_dataset(random_state=42)
dataset_name = 'Iris'
nb_cross_validation = 4
experiment_title = 'IrisClassification2'
total_budget = 150000
max_budget_per_config = 600


mlp_experiment(experiment_title, x_train, t_train, x_test, t_test, total_budget,
               max_budget_per_config, dataset_name, nb_cross_validation)
