"""
    @file:              DataGen.py
    @Author:            Alexandre Ayotte
    @Creation Date:     29/09/2019
    @Last modification: 29/09/2019
    @Description:       This program generate random data for toy problem that we want to solve with autoML method
"""

import sklearn
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:

    def __init__(self, train, test, model_name):
        self.train_size = train
        self.test_size = test
        self.model = model_name
