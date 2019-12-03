"""
    @file:              ResultManagement.py
    @Author:            Nicolas Raymond
    @Creation Date:     09/11/2019
    @Last modification: 14/11/2019
    @Description:       This file is dedicated to all result managing functions.
"""

import matplotlib.pyplot as plt
import csv
import os
import os.path


class ExperimentAnalyst:

    def __init__(self, tuning_method, model_name):

        """
        Class that generates intelligent and useful storage and visualization
        methods for all hyper-parameter tuning results.

        :param tuning_method: Name of the method used for hyper-parameter tuning
        """

        self.model_name = model_name
        self.tuning_method = tuning_method
        self.method_type = None               # Only used when tuning method is a gaussian process
        self.nbr_of_cross_validation = 1
        self.validation_size = 0.2
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

    def plot_accuracy_history(self, best_accuracy=False, show_plot=True):

        """
        Plots curve associated to loss history

        :return: Plot of loss
        """
        # If we want to see best loss history
        if best_accuracy:
            plt.plot(range(len(self.accuracy_history)), self.best_accuracy_history, color='b')
            plt.ylabel('best accuracy')

        else:
            plt.plot(range(len(self.accuracy_history)), self.accuracy_history, color='b')
            plt.ylabel('accuracy')

        plt.suptitle(self.tuning_method)
        plt.xlabel('iteration')

        if show_plot:
            plt.show()

        return plt

    def save_all_results(self, path, experiment_title, dataset_name, training_size, test_accuracy, noise=None):

        """
        Saves all results saved by the ExperimentAnalyst

        :param path: string representing the path that will contain the folder of the file
        :param experiment_title: string with title of the experiment
        :param dataset_name: string with the name of the dataset
        :param training_size: int indicating number of elements in training data set
        :param noise: noise added to the data set
        :param test_accuracy: accuracy obtained with test data set
        """

        # We initialize a list with all the folder needed to save the results
        folders_name = [experiment_title, self.tuning_method]

        # We adjust add an element to the list if there's a method type
        if self.method_type is not None:
            folders_name.append(self.method_type)

        # We create all folder expected in the folder Results (if they don't already exist)
        for folder_name in [experiment_title, self.tuning_method]:
            path = os.path.join(path, folder_name.upper(), '')
            self.create_folder(path)

        # We save all important data for the ExperimentAnalyst in the path concerned
        self.__save_tuning_summary(path, experiment_title, dataset_name, training_size, noise, test_accuracy)
        self.__save_accuracy_history(path)
        self.__save_accuracy_history(path, best_accuracy=True)
        self.__save_hyperparameters(path)

    def __save_accuracy_history(self, path, best_accuracy=False, save_plot=True):

        """
        Saves accuracy history in a csv file at the path indicated

        :param path: string representing the path that will contain the file
        :param best_accuracy: boolean that indicates if we want the best accuracy history
        :param save_plot: boolean indicating if we wish to save the accuracy plot
        """
        # Save data in csv file
        if best_accuracy:
            self.write_csv(path, 'best_acc_hist', self.best_accuracy_history)
            plot_path = path+'best_acc_plot'
        else:
            self.write_csv(path, 'acc_hist', self.accuracy_history)
            plot_path = path+'acc_plot'

        # Save plot of accuracy
        if save_plot:
            plot = self.plot_accuracy_history(best_accuracy, False)
            plot.savefig(plot_path)
            plot.clf()

    def __save_hyperparameters(self, path):

        """
        Saves all hyper-parameters evaluated in the tuning process

        :param path: string representing the path that will contain the file
        """

        self.write_csv(path, 'hyperparameters_hist', self.hyperparameters_history)

    def __save_tuning_summary(self, path, experiment_title, dataset_name, training_size, noise, test_accuracy):

        """
        Saves a summary of the tuning results in a .txt file

        :param path: string representing the path that will contain the file
        :param experiment_title: string with title of the experiment
        :param dataset_name: string with the name of the dataset
        :param training_size: int indicating number of elements in training data set
        :param noise: noise added to the data set
        :param test_accuracy: accuracy obtained with test data set
        """

        # We open the file
        f = open(path+'tuning_summary.txt', "w+")

        # We write the highlights
        f.write("Experiment title: %s \n\n" % experiment_title)
        f.write("Model name : %s \n\n" % self.model_name)
        f.write("Nbr. of cross validation done in each iteration : %g \n\n" % self.nbr_of_cross_validation)
        f.write("Validation size in cross validation : %g \n\n" % self.validation_size)
        f.write("Dataset name : %s \n\n" % dataset_name)
        f.write("Size of training set : %g \n\n" % training_size)

        if noise is not None:
            f.write("Noise : %g \n\n" % noise)

        f.write("Number of iterations : %g \n\n" % len(self.accuracy_history))
        f.write("Best accuracy obtained in tuning : %g \n\n" % self.actual_best_accuracy)
        f.write("Best hyper-parameters found : %s \n\n" % str(self.best_hyperparameters))
        f.write("Test accuracy : %g" % test_accuracy)

        # We close the file
        f.close()

    def reset(self):

        """
        Resets ExperimentAnalyst to ignition values
        """

        self.__init__(self.tuning_method, self.model_name)

    @staticmethod
    def write_csv(path, file_name, rows):

        """
        Generic method to write rows of data in an existing or new csv file.

        :param path: string representing the path that will contain the file
        :param file_name: name of the future csv file
        :param rows: list where each element will be a row in the csv file
        """

        complete_path = path+file_name+'.csv'
        with open(complete_path, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(rows)

        csvFile.close()

    @staticmethod
    def create_folder(directory):

        """
        Creates a folder with the directory mentioned

        :param directory: String that mention path where the folder is going to be created
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)

        except OSError:
            print('Error while creating directory : ' + directory)
