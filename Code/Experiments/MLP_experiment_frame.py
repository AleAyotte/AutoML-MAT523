"""
    @file:              MLP_experiment_frame.py
    @Author:            Nicolas Raymond
    @Creation Date:     10/12/2019
    @Last modification: 10/12/2019

    @Description:       Definition of a general method for MLP experiment
"""


# Import code needed
import sys
import os
from numpy import linspace
import pickle


# Append path of module to sys and import module
module_path = os.path.dirname(os.getcwd())
sys.path.append(module_path)
import Model as mod
from HPtuner import HPtuner, ContinuousDomain, DiscreteDomain


def mlp_experiment(experiment_title, x_train, t_train, x_test, t_test,
                   total_budget, max_budget_per_config, dataset_name, nb_cross_validation=1, noise=None):

    print('Experiment in process..\n')

    # We compute training size
    train_size = len(x_train)
    search_space = {'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                    'learning_rate_init': ContinuousDomain(-8, 0, log_scaled=True),
                    'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int).tolist())),
                    'hidden_layers_number': DiscreteDomain(range(1, 21)),
                    'layers_size': DiscreteDomain(range(5, 51))}

    # We initialize an MLP with default hyper-parameters and 4 hidden layers of 20 neurons to classify our data
    # and test its performance on both training and test data sets
    mlp = mod.MLP(hidden_layers_number=4, layers_size=20, max_iter=1000)
    mlp.fit(x_train, t_train)
    print('Initial score :', mlp.score(x_test, t_test))

    """
    Random search
    """

    print('\n\n RANDOM SEARCH \n\n')

    # We do a deep copy of our MLP for the test, set the experiment title and save the path to save the results
    save = pickle.dumps(mlp)
    mlp_for_rs = pickle.loads(save)
    results_path = os.path.join(os.path.dirname(module_path), 'Results')


    # We initialize a tuner with random search method and set our search space
    rs_tuner = HPtuner(mlp_for_rs, 'random_search', total_budget=total_budget,
                       max_budget_per_config=max_budget_per_config)

    rs_tuner.set_search_space(search_space)

    # We execute the tuning and save the results
    rs_results = rs_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    rs_results.save_all_results(results_path, experiment_title, dataset_name,
                                train_size, mlp_for_rs.score(x_test, t_test), noise=noise)

    """
    # TPE (Tree-structured Parzen Estimator )
    """

    print('\n\n TPE \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with tpe method and set our search space
    mlp_for_tpe = pickle.loads(save)
    tpe_tuner = HPtuner(mlp_for_tpe, 'tpe', total_budget=total_budget,
                        max_budget_per_config=max_budget_per_config)

    tpe_tuner.set_search_space(search_space)

    # We execute the tuning and save the results
    tpe_results = tpe_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    tpe_results.save_all_results(results_path, experiment_title, dataset_name,
                                 train_size, mlp_for_tpe.score(x_test, t_test))

    """
    # Simulated Annealing
    """

    print('\n\n SIMULATED ANNEALING \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with tpe method and set our search space
    mlp_for_anneal = pickle.loads(save)
    anneal_tuner = HPtuner(mlp_for_anneal, 'annealing', total_budget=total_budget,
                           max_budget_per_config=max_budget_per_config)

    anneal_tuner.set_search_space(search_space)

    # We execute the tuning and save the results
    anneal_results = anneal_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    anneal_results.save_all_results(results_path, experiment_title, dataset_name,
                                    train_size, mlp_for_anneal.score(x_test, t_test), noise=noise)

    """
    # Standard GP with EI acquisition function
    """

    print('\n\n GP WITH EI \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
    mlp_for_GP = pickle.loads(save)
    GP_tuner = HPtuner(mlp_for_GP, 'gaussian_process', total_budget=total_budget,
                       max_budget_per_config=max_budget_per_config)

    GP_tuner.set_search_space(search_space)

    # We execute the tuning using default parameter for GP
    # ('GP' as method type, 5 initial points to evaluate before the beginning and 'EI' acquisition)
    GP_results = GP_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)

    # We save the results
    GP_results.save_all_results(results_path, experiment_title, dataset_name,
                                train_size, mlp_for_GP.score(x_test, t_test), noise=noise)

    """
    # Standard GP with MPI acquisition function
    """

    print('\n\n GP WITH MPI \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
    mlp_for_GP2 = pickle.loads(save)
    GP_tuner2 = HPtuner(mlp_for_GP2, 'gaussian_process', total_budget=total_budget,
                        max_budget_per_config=max_budget_per_config)

    GP_tuner2.set_search_space(search_space)

    # We execute the tuning using default parameter for GP except MPI acquisition
    # ('GP' as method type, 5 initial points to evaluate before the beginning and 'MPI' acquisition)
    GP_results2 = GP_tuner2.tune(x_train, t_train, nb_cross_validation=nb_cross_validation, acquisition_function='MPI')

    # We save the results
    GP_results2.save_all_results(results_path, experiment_title, dataset_name,
                                 train_size, mlp_for_GP2.score(x_test, t_test), noise=noise)

    """
    # Hyperband
    """

    print('\n\n HYPERBAND \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
    mlp_hb = pickle.loads(save)
    mlp_hb_tuner = HPtuner(mlp_hb, 'hyperband', total_budget=total_budget,
                           max_budget_per_config=max_budget_per_config)

    mlp_hb_tuner.set_search_space(search_space)

    # We execute the tuning and save the results
    hb_results = mlp_hb_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    hb_results.save_all_results(results_path, experiment_title, dataset_name,
                                train_size, mlp_hb.score(x_test, t_test), noise=noise)

    """
    # Grid search
    """

    print('\n\n GRID SEARCH \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with the grid_search method and set our search space
    mlp_for_gs = pickle.loads(save)
    gs_tuner = HPtuner(mlp_for_gs, 'grid_search')
    gs_tuner.set_search_space({'alpha': DiscreteDomain(list(linspace(10 ** -8, 1, 5))),
                               'learning_rate_init': DiscreteDomain(list(linspace(10 ** -8, 1, 5))),
                               'batch_size': DiscreteDomain([200]),
                               'hidden_layers_number': DiscreteDomain([1, 5, 10, 15, 20]),
                               'layers_size': DiscreteDomain([20, 50])})

    # We execute the tuning and save the results
    gs_results = gs_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    gs_results.save_all_results(results_path, experiment_title, dataset_name,
                                train_size, mlp_for_gs.score(x_test, t_test), noise=noise)
