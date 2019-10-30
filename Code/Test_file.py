
from numpy import linspace
import Code.DataManager as dm
import Code.Model as mod
from Code.HPtuner import HPtuner, ContinuousDomain, DiscreteDomain


def main():

    test = "random_search"

    # We generate data for our tests
    dgen = dm.DataGenerator(200,200,"circles")
    x_train, t_train, x_test, t_test = dgen.generate_data(0.08)
    #dm.plot_data(x_train,t_train)


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
        print(mlp_tuner.search_space.space)
        mlp_tuner.tune(x_train, t_train, n_evals=400)

        # We look at the new results
        mlp.fit(x_train, t_train)
        mlp.plot_data(x_test, t_test)


if __name__ == '__main__':
    main()