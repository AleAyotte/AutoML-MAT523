"""
    @file:              ResultManagement.py
    @Author:            Nicolas Raymond
    @Creation Date:     09/11/2019
    @Last modification: 09/11/2019
    @Description:       This file is dedicated to all result managing functions.
"""

import matplotlib.pyplot as plt


class ExperimentAnalyst:

    """
    Class that generates intelligent and useful storage and visualization
    methods for all hyper-parameter tuning results.
    """

    def __init__(self, tuning_method):

        self.tuning_method = tuning_method
        self.hyperparameters_history = []
        self.best_hyperparameters = {}
        self.loss_history = []
        self.best_loss_history = []
        self.actual_best_loss = 1  # Worst possible loss

    def update(self, new_loss, hyperparams):

        """
        Updates all attributes of the ExperimentAnalyst considering the new loss

        :param new_loss: new loss obtained
        :param hyperparams: hyper-parameters associated to the new loss
        """

        # Update history
        self.loss_history.append(new_loss)
        self.hyperparameters_history.append(hyperparams)

        # Update best pair of hyper-parameters and loss if the actual one is beaten
        if new_loss < self.actual_best_loss:
            self.actual_best_loss = new_loss
            self.best_loss_history.append(new_loss)
            self.best_hyperparameters = hyperparams

        else:
            self.best_loss_history.append(self.actual_best_loss)

    def plot_loss_history(self, best_loss=False):

        """
        Plots curve associated to loss history

        :return: Plot of loss
        """
        # If we want to see best loss history
        if best_loss:
            plt.plot(range(1, len(self.loss_history) + 1), self.best_loss_history, color='b')
            plt.ylabel('best loss')

        else:
            plt.plot(range(1, len(self.loss_history)+1), self.loss_history, color='b')
            plt.ylabel('loss')

        plt.suptitle(self.tuning_method)
        plt.xlabel('iteration')
        plt.show()
