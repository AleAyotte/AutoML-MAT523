"""
    @file:              ResultManagement.py
    @Author:            Nicolas Raymond
    @Creation Date:     09/11/2019
    @Last modification: 09/11/2019
    @Description:       This file is dedicated to all result managing functions.
"""

import matplotlib.pyplot as plt


class ExperimentAnalyst:

    def __init__(self, tuning_method):

        """
        Class that generates intelligent and useful storage and visualization
        methods for all hyper-parameter tuning results.

        :param tuning_method: Name of the method used for hyper-parameter tuning
        """

        self.tuning_method = tuning_method
        self.hyperparameters_history = []
        self.best_hyperparameters = {}
        self.accuracy_history = []
        self.best_accuracy_history = []
        self.actual_best_accuracy = 0  # Worst possible accuracy

    def update(self, new_loss, hyperparams):

        """
        Updates all attributes of the ExperimentAnalyst considering the new loss

        :param new_loss: new loss obtained
        :param hyperparams: hyper-parameters associated to the new loss
        """

        # Update history
        accuracy = 1 - new_loss
        self.accuracy_history.append(accuracy)
        self.hyperparameters_history.append(hyperparams)

        # Update best pair of hyper-parameters and loss if the actual one is beaten
        if accuracy > self.actual_best_accuracy:
            self.actual_best_accuracy = accuracy
            self.best_accuracy_history.append(accuracy)
            self.best_hyperparameters = hyperparams

        else:
            self.best_accuracy_history.append(self.actual_best_accuracy)

    def plot_accuracy_history(self, best_accuracy=False):

        """
        Plots curve associated to loss history

        :return: Plot of loss
        """
        # If we want to see best loss history
        if best_accuracy:
            plt.plot(range(1, len(self.accuracy_history) + 1), self.best_accuracy_history, color='b')
            plt.ylabel('best accuracy')

        else:
            plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history, color='b')
            plt.ylabel('accuracy')

        plt.suptitle(self.tuning_method)
        plt.xlabel('iteration')
        plt.show()
