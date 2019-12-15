"""
    @file:              MLP_experiment_frame.py
    @Author:            Nicolas Raymond
    @Creation Date:     10/12/2019
    @Last modification: 12/12/2019

    @Description:       Definition of a general method for experiment
"""


# Import code needed
import sys
import os
import pickle


# Append path of module to sys and import module
module_path = os.path.dirname(os.getcwd())
sys.path.append(module_path)
from HPtuner import HPtuner


def run_experiment(model, experiment_title, x_train, t_train, x_test, t_test, search_space, grid_search_space,
                   total_budget, max_budget_per_config, dataset_name, nb_cross_validation=1, noise=None):

    print('\nExperiment in process..\n')

    # We compute training size
    train_size = len(x_train)

    model.fit(x_train, t_train)
    print('Initial score :', model.score(x_test, t_test))

    # We save a deep copy of our model for the tests, and save the results path
    save = pickle.dumps(model)
    results_path = os.path.join(os.path.dirname(module_path), 'Results')

    """
    #Random search
    """

    print('\n\n RANDOM SEARCH \n\n')

    # We do a deep copy of our MLP for the test

    # We do a deep copy of our MLP for the test, initialize a tuner with random search method and set our search space
    model_for_rs = pickle.loads(save)
    rs_tuner = HPtuner(model_for_rs, 'random_search', total_budget=total_budget,
                       max_budget_per_config=max_budget_per_config)

    rs_tuner.set_search_space(search_space)

    # We execute the tuning and save the results
    rs_results = rs_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    rs_results.save_all_results(results_path, experiment_title, dataset_name,
                                train_size, model_for_rs.score(x_test, t_test), noise=noise)

    """
    # TPE (Tree-structured Parzen Estimator )
    """

    print('\n\n TPE \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with tpe method and set our search space
    model_for_tpe = pickle.loads(save)
    tpe_tuner = HPtuner(model_for_tpe, 'tpe', total_budget=total_budget,
                        max_budget_per_config=max_budget_per_config)

    tpe_tuner.set_search_space(search_space)

    # We execute the tuning and save the results
    tpe_results = tpe_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    tpe_results.save_all_results(results_path, experiment_title, dataset_name,
                                 train_size, model_for_tpe.score(x_test, t_test))

    """
    # Simulated Annealing
    """

    print('\n\n SIMULATED ANNEALING \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with tpe method and set our search space
    model_for_anneal = pickle.loads(save)
    anneal_tuner = HPtuner(model_for_anneal, 'annealing', total_budget=total_budget,
                           max_budget_per_config=max_budget_per_config)

    anneal_tuner.set_search_space(search_space)

    # We execute the tuning and save the results
    anneal_results = anneal_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    anneal_results.save_all_results(results_path, experiment_title, dataset_name,
                                    train_size, model_for_anneal.score(x_test, t_test), noise=noise)

    """
    # Standard GP with EI acquisition function
    """

    print('\n\n GP WITH EI \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
    model_for_GP = pickle.loads(save)
    GP_tuner = HPtuner(model_for_GP, 'gaussian_process', total_budget=total_budget,
                       max_budget_per_config=max_budget_per_config)

    GP_tuner.set_search_space(search_space)

    # We execute the tuning using default parameter for GP
    # ('GP' as method type, 5 initial points to evaluate before the beginning and 'EI' acquisition)
    GP_results = GP_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)

    # We save the results
    GP_results.save_all_results(results_path, experiment_title, dataset_name,
                                train_size, model_for_GP.score(x_test, t_test), noise=noise)

    """
    # Standard GP with MPI acquisition function
    """

    print('\n\n GP WITH MPI \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
    model_for_GP2 = pickle.loads(save)
    GP_tuner2 = HPtuner(model_for_GP2, 'gaussian_process', total_budget=total_budget,
                        max_budget_per_config=max_budget_per_config)

    GP_tuner2.set_search_space(search_space)

    # We execute the tuning using default parameter for GP except MPI acquisition
    # ('GP' as method type, 5 initial points to evaluate before the beginning and 'MPI' acquisition)
    GP_results2 = GP_tuner2.tune(x_train, t_train, nb_cross_validation=nb_cross_validation, acquisition_function='MPI')

    # We save the results
    GP_results2.save_all_results(results_path, experiment_title, dataset_name,
                                 train_size, model_for_GP2.score(x_test, t_test), noise=noise)

    """
    # Hyperband
    """

    print('\n\n HYPERBAND \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with the standard GP method and set our search space
    model_for_hb = pickle.loads(save)
    model_for_hb_tuner = HPtuner(model_for_hb, 'hyperband', total_budget=total_budget,
                                 max_budget_per_config=max_budget_per_config)

    model_for_hb_tuner.set_search_space(search_space)

    # We execute the tuning and save the results
    hb_results = model_for_hb_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    hb_results.save_all_results(results_path, experiment_title, dataset_name,
                                train_size, model_for_hb.score(x_test, t_test), noise=noise)

    """
    # Grid search
    """

    print('\n\n GRID SEARCH \n\n')

    # We do a deep copy of our MLP for the test, initialize a tuner with the grid_search method and set our search space
    model_for_gs = pickle.loads(save)
    gs_tuner = HPtuner(model_for_gs, 'grid_search')
    gs_tuner.set_search_space(grid_search_space)

    # We execute the tuning and save the results
    gs_results = gs_tuner.tune(x_train, t_train, nb_cross_validation=nb_cross_validation)
    gs_results.save_all_results(results_path, experiment_title, dataset_name,
                                train_size, model_for_gs.score(x_test, t_test), noise=noise)
