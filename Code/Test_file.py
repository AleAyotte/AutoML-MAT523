
from numpy import linspace
import Code.DataManager as dm
import Code.Model as mod
from Code.HPtuner import HPtuner, ContinuousDomain, DiscreteDomain


def main():

    test = "tpe"

    # We generate data for our tests
    dgen = dm.DataGenerator(500, 500, "nSpiral")
    noise = 0.28
    x_train, t_train, x_test, t_test = dgen.generate_data(0.28, 5)
    dm.plot_data(x_train, t_train)

    """
    
    GRID SEARCH TEST
    
    """
    if test == 'grid_search':

        # We generate an MLP to classify our data (5 hidden layers of 20 neurons)
        mlp = mod.MLP((20, 20, 20, 20, 20), max_iter=1000)

        # We optimize hyper-parameters with grid_search
        mlp_tuner = HPtuner(mlp, 'grid_search', test_default_hyperparam=True)
        mlp_tuner.set_search_space({'alpha': DiscreteDomain(list(linspace(0, 1, 20))),
                                   'learning_rate_init': DiscreteDomain(list(linspace(0.0001, 1, 20)))})
        results = mlp_tuner.tune(x_train, t_train)

        # We look at the tuning results
        results.plot_accuracy_history()
        results.plot_accuracy_history(best_accuracy=True)
        mlp.plot_data(x_test, t_test)

    """
    
    RANDOM SEARCH TEST
    
    """
    if test == 'random_search':

        # We generate an MLP to classify our data (5 hidden layers of 20 neurons)
        mlp = mod.MLP((20, 20, 20, 20, 20), max_iter=1000)

        # We optimize hyper-parameters with random search
        mlp_tuner = HPtuner(mlp, 'random_search', test_default_hyperparam=True)
        mlp_tuner.set_search_space({'alpha': ContinuousDomain(0, 1),
                                    'learning_rate_init': ContinuousDomain(-6, 0, log_scaled=True)})

        results = mlp_tuner.tune(x_train, t_train, n_evals=25, nb_cross_validation=2)

        # We look at the new results
        results.plot_accuracy_history()
        results.plot_accuracy_history(best_accuracy=True)
        mlp.plot_data(x_test, t_test)

    """
    TPE TEST 
    
    """

    if test == 'tpe':

        # We generate an MLP to classify our data (5 hidden layers of 20 neurons)
        mlp = mod.MLP((20, 20, 20, 20, 20), max_iter=1000)

        # We optimize hyper-parameters with random search
        mlp_tuner = HPtuner(mlp, 'tpe', test_default_hyperparam=True)
        mlp_tuner.set_search_space({'alpha': ContinuousDomain(-8, 0, log_scaled=True),
                                    'learning_rate_init': ContinuousDomain(-6, 0, log_scaled=True),
                                    'batch_size': DiscreteDomain(linspace(50, 500, 10, dtype=int))})

        results = mlp_tuner.tune(x_train, t_train, n_evals=80, nb_cross_validation=2)

        # We look at the new results
        results.plot_accuracy_history()
        results.plot_accuracy_history(best_accuracy=True)
        mlp.plot_data(x_test, t_test)
        results.save_all_results('Dummy_MLP_5_20', dgen.model, dgen.train_size, noise)

        # ---------------------------------------------------------------------------------- #

        # We generate another set of data to test an svm with a polynomial kernel
        dgen = dm.DataGenerator(500, 500, "circles")
        noise = 0.05
        x_train, t_train, x_test, t_test = dgen.generate_data(noise)
        dm.plot_data(x_train, t_train)

        # We do the same exercice for an svm
        svm = mod.SVM(kernel='poly')
        svm_tuner = HPtuner(svm, 'tpe', test_default_hyperparam=True)
        svm_tuner.set_search_space({'C': ContinuousDomain(-2, 0, log_scaled=True),
                                    'degree': DiscreteDomain([1, 2, 3, 4, 5, 6])})
        results = svm_tuner.tune(x_train, t_train, n_evals=500, nb_cross_validation=4)
        results.plot_accuracy_history()
        results.plot_accuracy_history(best_accuracy=True)
        svm.plot_data(x_test, t_test)
        results.save_all_results('Dummy_polynomial500', dgen.model, dgen.train_size, noise)

    """
    
    GAUSSIAN PROCESS TEST 
    
    """

    if test == 'gaussian_process':

        # We generate an MLP to classify our data (5 hidden layers of 20 neurons)
        mlp = mod.MLP((20, 20, 20, 20, 20), max_iter=1000)

        # We optimize hyper-parameters with random search
        mlp_tuner = HPtuner(mlp, 'gaussian_process', test_default_hyperparam=True)
        mlp_tuner.set_search_space({'alpha': ContinuousDomain(-6, 0, log_scaled=True),
                                    'learning_rate_init': ContinuousDomain(-6, 0, log_scaled=True),
                                    'batch_size': DiscreteDomain(list(linspace(50, 500, 10, dtype=int)))})

        results = mlp_tuner.tune(x_train, t_train, n_evals=50, nb_cross_validation=2)

        # We look at the tuning results
        results.plot_accuracy_history()
        results.plot_accuracy_history(best_accuracy=True)
        mlp.plot_data(x_test, t_test)
        results.save_all_results('Dummy_MLP_5_20_500', dgen.model, dgen.train_size, noise)


if __name__ == '__main__':
    main()
