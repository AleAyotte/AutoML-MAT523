"""
    @file:              nSpiralExperiment1.py
    @Author:            Nicolas Raymond
    @Creation Date:     18/11/2019
    @Last modification: 18/11/2019
    @Description:       For this first experiment, we will evaluate the performance of all hyper-parameter optimization
                        methods implemented in a simple context with a fixed budget of 500 evaluations. More precisely,
                        considering a simple 2D points classification problem called nSpiral with 5 classes,
                        800 training points and 800 test points generated and a value of 0.40 as the standard deviation
                        of the Gaussian noise added to the data, will we initialize a Sklearn MLP with 4 hidden layers
                        of 20 neurons with default parameter and try to find the best values for  𝛼
                        (L2 penalty (regularization term) parameter), learning rate init (initial learning rate used),
                        𝛽1  (exponential decay rate for estimates of first moment vector in adam) and finaly  𝛽2
                        (exponential decay rate for estimates of second moment vector in adam) with all methods
                        available.
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

# We generate data for our tests and two global variables for all test
dgen = dm.DataGenerator(800, 800, "nSpiral")
noise = 0.40
x_train, t_train, x_test, t_test = dgen.generate_data(noise, 5, seed=10512)
nb_cross_validation = 2
nb_evals = 500

# We initialize an MLP with default hyper-parameters and 3 hidden layers of 20 neurons to classify our data
# and test its performance on both training and test data sets
mlp = mod.MLP(hidden_layers_number=4, layers_size=20, max_iter=1000)
mlp.fit(x_train, t_train)
print(mlp.score(x_test, t_test))

"""
Random search
"""

# We do a deep copy of our MLP for the test, set the experiment title and save the path to save the results
save = pickle.dumps(mlp)
mlp_for_rs = pickle.loads(save)
experiment_title = 'nSPIRAL1'
results_path = os.path.join(os.path.dirname(module_path), 'Results')

# We initialize a tuner with random search method and set our search space
rs_tuner = HPtuner(mlp_for_rs, 'random_search')
rs_tuner.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                           'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                           'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int)))})

# We execute the tuning and save the results
rs_results = rs_tuner.tune(x_train, t_train, n_evals=nb_evals, nb_cross_validation=nb_cross_validation)
rs_results.save_all_results(results_path, experiment_title, dgen.model,
                            dgen.train_size, noise, mlp_for_rs.score(x_test, t_test))

"""
TPE (Tree-structured Parzen Estimator )
"""

# We do a deep copy of our MLP for the test, initialize a tuner with tpe method and set our search space
mlp_for_tpe = pickle.loads(save)
tpe_tuner = HPtuner(mlp_for_tpe, 'tpe')
tpe_tuner.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                            'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                            'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int)))})

# We execute the tuning and save the results
tpe_results = tpe_tuner.tune(x_train, t_train, n_evals=nb_evals, nb_cross_validation=nb_cross_validation)
tpe_results.save_all_results(results_path, experiment_title, dgen.model,
                             dgen.train_size, noise, mlp_for_tpe.score(x_test, t_test))

"""
Standard GP with EI acquisition function
"""

# We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
mlp_for_GP = pickle.loads(save)
GP_tuner = HPtuner(mlp_for_GP, 'gaussian_process')
GP_tuner.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                           'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                           'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int)))})

# We execute the tuning using default parameter for GP
# ('GP' as method type, 5 initial points to evaluate before the beginning and 'EI' acquisition)
GP_results = GP_tuner.tune(x_train, t_train, n_evals=nb_evals, nb_cross_validation=nb_cross_validation)

# We save the results
GP_results.save_all_results(results_path, experiment_title, dgen.model,
                            dgen.train_size, noise, mlp_for_GP.score(x_test, t_test))

"""
Standard GP with MPI acquisition function
"""
# We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
mlp_for_GP2 = pickle.loads(save)
GP_tuner2 = HPtuner(mlp_for_GP2, 'gaussian_process')
GP_tuner2.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                            'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                            'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int)))})

# We execute the tuning using default parameter for GP except MPI acquisition
# ('GP' as method type, 5 initial points to evaluate before the beginning and 'MPI' acquisition)
GP_results2 = GP_tuner2.tune(x_train, t_train, n_evals=nb_evals,
                             nb_cross_validation=nb_cross_validation, acquisition_function='MPI')

# We save the results
GP_results2.save_all_results(results_path, experiment_title, dgen.model,
                             dgen.train_size, noise, mlp_for_GP2.score(x_test, t_test))

"""
Grid search
"""

# We do a deep copy of our MLP for the test, initialize a tuner with the grid_search method and set our search space
mlp_for_gs = pickle.loads(save)
gs_tuner = HPtuner(mlp_for_gs, 'grid_search')
gs_tuner.set_search_space({'alpha': DiscreteDomain(list(linspace(10 ** -8, 1, 10, dtype=int))),
                           'learning_rate_init': DiscreteDomain(list(linspace(10 ** -8, 1, 10, dtype=int))),
                           'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int)))})

# We execute the tuning and save the results
gs_results = gs_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
gs_results.save_all_results(results_path, experiment_title, dgen.model,
                            dgen.train_size, noise, mlp_for_gs.score(x_test, t_test))

