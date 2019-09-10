"""
 *  DataGenerator.py
 *  @author: Alexandre Ayotte - stagiaire en intelligence artificielle. Soucy.inc
 *  @version:   0.01 - 2019-05-29
 *
 *  Description:
 *      Ce programme a pour but de générer différent ensemble de données aléatoires. Comme des ensembles de type
 *      N-Circle ou encore des nuages de points à n-classes. Il s'agit bien sûr d'un programme d'entrainement pour
 *      les données en IA. Cette classe fait partie d'un projet visant à apprendre tensorflow. Toutefois, le module
 *      tensorflow ne sera pas utilisé dans ce programme. L'objectif étant de le rendre le plus maléable possible.
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, nb_train, nb_test, nb_class):
        """
        Classe qui génère des données aléatoire selon plusieurs modèles disponibles(2).

        :param nb_train: Nombre de données d'entrainement
        :param nb_test:  Nombre de données de test
        :param nb_classes: Nombre de classes
        """
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.nb_class = nb_class

    def polar_to_cart(self, radius, angle):
        """
        Convertie les coordonnées polaires en coordonnées cartésiennes

        :param radius: un array contenant les rayons de chacun des points
        :param angle: un array contenant l'angle de chacun des points
        :return: deux array contenant les coordonnées en horizontales et verticales respectivement
        """

        axis_x = radius * np.cos(angle)
        axis_y = radius * np.sin(angle)
        return axis_x.astype(dtype='float32'), axis_y.astype(dtype='float32')

    def n_circle(self, nb_data, noise, distance):
        """
        Génère des données alatoires de type N-circle où N représente le nombre de classes. Les données sont
        répartie en coordonnées polaire selon une distribution normale sur la rayon.

        :param nb_data: Nombre de données à générer au total.
        :param noise: Écart-type de la distribution normale qui sera utilisée pour générer les données
        :param distance: distance minimale entre les espérances de chacune des classes.
        :return: Deux array, un contenant données en coordonnées cartésiennes et l'autre les classes correspondantes.
        """

        classes = np.array([])
        radius = np.array([])

        angle = 2*math.pi*np.random.rand(nb_data)

        # Donnée restante à générer
        data_left = nb_data

        for i in range(self.nb_class):
            # On sépare les données de la manière la plus équitable possible
            split = round(data_left/ (self.nb_class - i))

            # On génère un vecteur de rayon aléatoire pour cette classe
            temp = np.random.normal(size=split, loc=distance*(i + 1), scale=noise)
            data_left -= split
            radius = np.append(radius, temp)
            classes = np.append(classes, np.ones(split) * i)

        data = np.vstack([self.polar_to_cart(radius, angle)]).T
        return data, classes.astype(dtype='int32')

    def n_region(self, nb_data, noise):
        """
        Génère des données alatoires de type N-region où N représente le nombre de classes. Les données sont
        répartie en coordonnées polaire selon une distribution normale sur l'angle. Les données sont séparées
        en trois triangles pointant tous vers le centre.

        :param nb_data: Nombre de données aléatoires à générer
        :param noise: Écart-type de la distribution normale qui sera utilisée pour générer les données
        :return: Deux array, un contenant données en coordonnées cartésiennes et l'autre les classes correspondantes.
        """

        classes = np.array([])
        angle = np.array([])

        radius = 4 * np.random.rand(nb_data) + 1
        data_left = nb_data

        for i in range(self.nb_class):
            # On sépare les données de la manière la plus équitable possible
            split = round(data_left / (self.nb_class - i))

            # On génère un vecteur de rayon aléatoire pour cette classe
            temp = np.random.normal(size=split, loc=(2*math.pi*i /self.nb_class), scale=noise)
            data_left -= split
            angle = np.append(angle, temp)
            classes = np.append(classes, np.ones(split) * i)

        data = np.vstack([self.polar_to_cart(radius, angle)]).T
        return data, classes.astype(dtype='int32')

    def n_spike(self, nb_data, noise, distance, amplitude):
        """
               Génère des données alatoires de type N-circle où N représente le nombre de classes. Les données sont
               répartie en coordonnées polaire selon une distribution normale sur la rayon.

               :param nb_data: Nombre de données à générer au total.
               :param noise: Écart-type de la distribution normale qui sera utilisée pour générer les données
               :param distance: Distance minimale entre les espérances de chacune des classes.
               :param amplitude: Amplitude sinusoïdale.
               :return: Deux array, un contenant données en coordonnées cartésiennes et l'autre les classes correspondantes.
               """

        classes = np.array([])
        radius = np.array([])

        angle = 2 * math.pi * np.random.rand(nb_data)
        data_left = nb_data

        for i in range(self.nb_class):
            # On sépare les données de la manière la plus équitable possible
            split = round(data_left / (self.nb_class - i))

            # On calcul la distance par rapport au centre.
            mean = distance * (i + 1)

            # On génère un vecteur de rayon aléatoire pour cette classe
            temp = np.random.normal(size=split, loc=mean, scale=noise)
            data_left -= split
            radius = np.append(radius, temp)
            classes = np.append(classes, np.ones(split) * i)
            np.set_printoptions(precision=3)

        radius += amplitude * np.sin(5*angle)
        data = np.vstack([self.polar_to_cart(radius, angle)]).T
        return data, classes.astype(dtype='int32')

    def generer_donnees(self, modele='nCircle', noise=0.5, distance=5, amplitude=1):
        """
        Génère des données d'entrainement et de test selon un modèle donnée. Pour l'instant, les choix sont
        seulement nCircle et nRegion

        :param modele: Le modèle a utlisé pour générer les donnnées
        :param noise:  L'écart-type sur les données
        :param distance:  La distance entre les anneaux si l'on prend le modèle nCircle
        :param amplitude:   L'amplitude sinusoidale pour le modèle nSpike
        :return: un ensemble d'entrainement et un ensemble de test.
        """

        if modele == "nCircle":
            x_train, t_train = self.n_circle(self.nb_train, noise, distance)
            x_test, t_test = self.n_circle(self.nb_test, noise, distance)
        elif modele == "nRegion":
            x_train, t_train = self.n_region(self.nb_train, noise)
            x_test, t_test = self.n_region(self.nb_test, noise)
        elif modele == "nSpike":
            x_train, t_train = self.n_spike(self.nb_train, noise, distance, amplitude)
            x_test, t_test = self.n_spike(self.nb_test, noise, distance, amplitude)
        else:
            return -1
        return x_train, t_train, x_test, t_test


def afficher(data, classes):
    """
    Permet d'afficher les données dans un graphique convivial

    :param data: Une matrice Nx2 contenant les coordonnées cartésiennes à afficher.
    :param classes: Un array contenant les classes correspondantes aux données
    :return:
    """

    # colour = np.where(classes==0, 'r', np.where(classes==1, 'b', 'g'))
    plt.scatter(data[:, 0], data[:, 1], s=105, c=classes, edgecolors='b')
    plt.show()