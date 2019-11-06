
from numpy import linspace
import Code.DataManager as dm
import Code.Model as mod
from Code.HPtuner import HPtuner, ContinuousDomain, DiscreteDomain


def main():

    test = "gaussian_process"

    # We generate data for our tests
    dgen = dm.DataGenerator(500, 500, "nSpiral")
    x_train, t_train, x_test, t_test = dgen.generate_data(0.35)
    dm.plot_data(x_train,t_train)

    """
    
    GRID SEARCH TEST
    
    """
    if test == 'grid_search':

        # We generate an MLP to classify our data (5 hidden layers of 20 neurons)
        mlp = mod.MLP((20, 20, 20, 20, 20), max_iter=1000)

        # We train our model without hyper-parameter tuning
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)

        # We optimize hyper-parameters with grid_search
        mlp_tuner = HPtuner(mlp, 'grid_search')
        mlp_tuner.set_search_space({'alpha': DiscreteDomain(list(linspace(0, 1, 20))),
                                   'learning_rate_init': DiscreteDomain(list(linspace(0.0001, 1, 20)))})
        mlp_tuner.tune(x_train, t_train)

        # We look at the new results
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)

    """
    
    RANDOM SEARCH TEST
    
    """
    if test == 'random_search':

        # We generate an MLP to classify our data (5 hidden layers of 20 neurons)
        mlp = mod.MLP((20, 20, 20, 20, 20), max_iter=1000)

        # We train our model without hyper-parameter tuning
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)

        # We optimize hyper-parameters with random search
        mlp_tuner = HPtuner(mlp, 'random_search')
        mlp_tuner.set_search_space({'alpha': ContinuousDomain(0, 1),
                                    'learning_rate_init': ContinuousDomain(0, 1)})

        mlp_tuner.tune(x_train, t_train, n_evals=400)

        # We look at the new results
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)

    """
    TPE TEST 
    
    """

    if test == 'tpe':

        # We generate an MLP to classify our data (5 hidden layers of 20 neurons)
        mlp = mod.MLP((20, 20, 20, 20, 20), max_iter=1000)

        # We train our model without hyper-parameter tuning
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)

        # We optimize hyper-parameters with random search
        mlp_tuner = HPtuner(mlp, 'tpe')
        mlp_tuner.set_search_space({'alpha': ContinuousDomain(0, 1),
                                    'learning_rate_init': ContinuousDomain(0.0001, 1)})

        mlp_tuner.tune(x_train, t_train, n_evals=100, nb_cross_validation=4)

        # We look at the new results
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)

        # ---------------------------------------------------------------------------------- #

        # We generate another set of data to test an svm with a polynomial kernel
        dgen = dm.DataGenerator(500, 500, "circles")
        x_train, t_train, x_test, t_test = dgen.generate_data(0.08)
        dm.plot_data(x_train, t_train)

        # We do the same exercice for an svm
        svm = mod.SVM(kernel='poly')
        svm.fit(x_train, t_train)
        svm.plot_data(x_test, t_test)
        svm_tuner = HPtuner(svm, 'tpe')
        svm_tuner.set_search_space({'C': ContinuousDomain(0.0001, 1),
                                    'degree': DiscreteDomain([1, 2, 3, 4, 5, 6])})
        svm_tuner.tune(x_train, t_train, n_evals=100, nb_cross_validation=3)
        svm.fit(x_train, t_train)
        svm.plot_data(x_test, t_test)

    """
    
    GAUSSIAN PROCESS TEST 
    
    """

    if test == 'gaussian_process':

        # We generate an MLP to classify our data (5 hidden layers of 20 neurons)
        mlp = mod.MLP((20, 20, 20, 20, 20), max_iter=1000)

        # We train our model without hyper-parameter tuning
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)

        # We optimize hyper-parameters with random search
        mlp_tuner = HPtuner(mlp, 'gaussian_process')
        mlp_tuner.set_search_space({'alpha': ContinuousDomain(0, 1),
                                    'learning_rate_init': ContinuousDomain(0.0001, 0.05)})

        mlp_tuner.tune(x_train, t_train, n_evals=50)

        # We look at the new results
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)

        # We do the same exercice to test an svm with an rbf kernel
        svm = mod.SVM()
        svm.fit(x_train, t_train)
        svm.plot_data(x_test, t_test)
        svm_tuner = HPtuner(svm, 'gaussian_process')
        svm_tuner.set_search_space({'C': ContinuousDomain(0.0001, 1),
                                    'gamma': ContinuousDomain(0, 1)})
        svm_tuner.tune(x_train, t_train, n_evals=50)
        svm.fit(x_train, t_train)
        svm.plot_data(x_test, t_test)


if __name__ == '__main__':
    main()
