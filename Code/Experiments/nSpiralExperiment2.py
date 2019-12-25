"""
    @file:              nSpiralExperiment2.py
    @Author:            Nicolas Raymond
    @Creation Date:     10/12/2019
    @Last modification: 11/12/2019

    @Description:       For this experiment, we will evaluate the performances of all hyper-parameter optimization
                        methods implemented in a simple context with a fixed total budget of
                        (250x200x2 = 100 000 epochs), a max budget per config of 400 epochs and a number
                        of two cross validation per config tested. For a better understanding the budget will allow
                        our model to evaluate 250 different configurations with non-bandit optimization method.


                        Now, considering a simple 2D points classification problem called nSpiral with 5 classes,
                        500 training points and 500 test points generated and a value of 0.35 as the standard deviation
                        of the Gaussian noise added to the data, will we initialize a a Sklearn MLP with 4 hidden layers
                        of 20 neurons with default parameter and try to find the best values for alpha
                        (L2 penalty (regularization term) parameter), learning rate init (initial learning rate used),
                        beta1  (exponential decay rate for estimates of first moment vector in adam), beta2
                        (exponential decay rate for estimates of second moment vector in adam), hidden layers number
                        (between 1 and 20) and finally layers size (number of neurons on each hidden layer between
                        5 and 50) with all methods available.

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
dgen = dm.DataGenerator(500, 500, "nSpiral")
noise = 0.40
x_train, t_train, x_test, t_test = dgen.generate_data(noise, 5, seed=10512)
dataset_name = 'nSpiral'
experiment_title = 'SPIRAL2'
nb_cross_validation = 2
total_budget = 100000
max_budget_per_config = 400

mlp_experiment(experiment_title, x_train, t_train, x_test, t_test, total_budget, max_budget_per_config,
               dataset_name, nb_cross_validation, noise)
