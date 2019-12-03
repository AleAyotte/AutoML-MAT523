"""
    @file:              IrisExperiment.py
    @Author:            Nicolas Raymond
    @Creation Date:     02/12/2019
    @Last modification: 02/12/2019
    @Description:       For this experiment, we will evaluate the performance of all hyper-parameter optimization
                        methods implemented in a simple context with a fixed budget of 250 evaluations. More precisely,
                        considering Iris data set classification problem (150 instances, 4 attributes, 3 classes),
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
import pickle


# Append path of module to sys and import module
module_path = os.path.dirname(os.getcwd())
sys.path.append(module_path)
import DataManager as dm
import Model as mod
from HPtuner import HPtuner, ContinuousDomain, DiscreteDomain

# We generate data for our tests and global variables for all tests
x_train, t_train, x_test, t_test = dm.load_iris_dataset(random_state=42)
dataset = 'Iris'
train_size = len(x_train)
nb_cross_validation = 4
nb_evals = 250

# We initialize an MLP with default hyper-parameters and 3 hidden layers of 20 neurons to classify our data
# and test its performance on both training and test data sets
mlp = mod.MLP(hidden_layers_number=4, layers_size=20, max_iter=1000)
mlp.fit(x_train, t_train)
print(mlp.score(x_test, t_test))

"""
Random search
"""

print('\n\n RANDOM SEARCH \n\n')

# We do a deep copy of our MLP for the test, set the experiment title and save the path to save the results
save = pickle.dumps(mlp)
mlp_for_rs = pickle.loads(save)
experiment_title = 'IrisClassification'
results_path = os.path.join(os.path.dirname(module_path), 'Results')

# We initialize a tuner with random search method and set our search space
rs_tuner = HPtuner(mlp_for_rs, 'random_search')
rs_tuner.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                           'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                           'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int))),
                           'hidden_layers_number': DiscreteDomain(range(1, 21)),
                           'layers_size': DiscreteDomain(range(5, 51))})

# We execute the tuning and save the results
rs_results = rs_tuner.tune(x_train, t_train, n_evals=nb_evals, nb_cross_validation=nb_cross_validation)
rs_results.save_all_results(results_path, experiment_title, dataset,
                            train_size, mlp_for_rs.score(x_test, t_test))

"""
TPE (Tree-structured Parzen Estimator )
"""

print('\n\n TPE \n\n')


# We do a deep copy of our MLP for the test, initialize a tuner with tpe method and set our search space
mlp_for_tpe = pickle.loads(save)
tpe_tuner = HPtuner(mlp_for_tpe, 'tpe')
tpe_tuner.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                            'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                            'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int))),
                            'hidden_layers_number': DiscreteDomain(range(1, 21)),
                            'layers_size': DiscreteDomain(range(5, 51))})

# We execute the tuning and save the results
tpe_results = tpe_tuner.tune(x_train, t_train, n_evals=nb_evals, nb_cross_validation=nb_cross_validation)
tpe_results.save_all_results(results_path, experiment_title, dataset,
                             train_size, mlp_for_tpe.score(x_test, t_test))

"""
Standard GP with EI acquisition function
"""

print('\n\n GP WITH EI \n\n')


# We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
mlp_for_GP = pickle.loads(save)
GP_tuner = HPtuner(mlp_for_GP, 'gaussian_process')
GP_tuner.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                           'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                           'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int))),
                           'hidden_layers_number': DiscreteDomain(range(1, 21)),
                           'layers_size': DiscreteDomain(range(5, 51))})

# We execute the tuning using default parameter for GP
# ('GP' as method type, 5 initial points to evaluate before the beginning and 'EI' acquisition)
GP_results = GP_tuner.tune(x_train, t_train, n_evals=nb_evals, nb_cross_validation=nb_cross_validation)

# We save the results
GP_results.save_all_results(results_path, experiment_title, dataset,
                            train_size, mlp_for_GP.score(x_test, t_test))

"""
Standard GP with MPI acquisition function
"""

print('\n\n GP WITH MPI \n\n')

# We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
mlp_for_GP2 = pickle.loads(save)
GP_tuner2 = HPtuner(mlp_for_GP2, 'gaussian_process')
GP_tuner2.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                            'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                            'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int))),
                            'hidden_layers_number': DiscreteDomain(range(1, 21)),
                            'layers_size': DiscreteDomain(range(5, 51))})

# We execute the tuning using default parameter for GP except MPI acquisition
# ('GP' as method type, 5 initial points to evaluate before the beginning and 'MPI' acquisition)
GP_results2 = GP_tuner2.tune(x_train, t_train, n_evals=nb_evals,
                             nb_cross_validation=nb_cross_validation, acquisition_function='MPI')

# We save the results
GP_results2.save_all_results(results_path, experiment_title, dataset,
                             train_size, mlp_for_GP2.score(x_test, t_test))

"""
Grid search
"""

print('\n\n GRID SEARCH \n\n')

# We do a deep copy of our MLP for the test, initialize a tuner with the grid_search method and set our search space
mlp_for_gs = pickle.loads(save)
gs_tuner = HPtuner(mlp_for_gs, 'grid_search')
gs_tuner.set_search_space({'alpha': DiscreteDomain(list(linspace(10 ** -8, 1, 5, dtype=int))),
                           'learning_rate_init': DiscreteDomain(list(linspace(10 ** -8, 1, 5, dtype=int))),
                           'batch_size': DiscreteDomain([200]),
                           'hidden_layers_number': DiscreteDomain([1, 5, 10, 15, 20]),
                           'layers_size': DiscreteDomain([20, 50])})

# We execute the tuning and save the results
gs_results = gs_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
gs_results.save_all_results(results_path, experiment_title, dataset,
                            train_size, mlp_for_gs.score(x_test, t_test))
