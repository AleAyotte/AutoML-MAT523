"""
    @file:              Main.py
    @Author:            Alexandre Ayotte
                        Nicolas Raymond
    @Creation Date:     27/11/2019
    @Last modification: 15/12/2019
    @Description:       Experiment file on the CIFAR10 Dataset using residual network.
"""

# Import code needed
import sys
import os
import numpy as np

# Append path of module to sys and import module
sys.path.append(os.getcwd())
module_path = os.path.dirname(os.getcwd())
sys.path.append(module_path)
import DataManager as dm
import Model as mod
from HPtuner import HPtuner, ContinuousDomain, DiscreteDomain
from Experiment_frame import run_experiment

# Generate training data
d_train, d_test = dm.load_cifar10()
input_size = np.array([32, 32, 3])

# Set global variables
dataset_name = 'CIFAR10'
nb_cross_validation = 1
experiment_title = 'CF10Exp1'
total_budget = 10000
max_budget_per_config = 200
train_size = 5000

# ------------------------------------------------------------------------------------------
#                                           RESNET
# ------------------------------------------------------------------------------------------

conv = np.array([16, 3, 1])
res_config = np.array([[3, 3], [3, 3], [3, 3]])
pool1 = np.array([0, 0, 0])  # No pooling layer after the first convolution
pool2 = np.array([4, 1, 1])  # Average pooling after the first convolution layer.
fc_nodes = None

net = mod.SimpleResNet(num_classes=10, num_res=3, activation='relu', version=1, input_dim=input_size, lr=0.021457992,
                       alpha=0.001149, eps=0.1, b_size=100, num_epoch=200, num_stop_epoch=20, lr_decay_rate=10,
                       num_lr_decay=4, valid_size=0.05, tol=0.004, save_path="checkpoint.pth")

print(net)


search_space = {'lr': ContinuousDomain(-7, -1, log_scaled=True),
                'alpha': ContinuousDomain(-10,  -1, log_scaled=True),
                'eps': DiscreteDomain([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]),
                'b_size': DiscreteDomain(np.arange(50, 360, 10).tolist()),
                'num_res': DiscreteDomain([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                'lr_decay_rate': DiscreteDomain(np.arange(2, 40, 1).tolist()),
                'activation': DiscreteDomain(['elu', 'relu', 'swish', 'mish']),
                'version': DiscreteDomain([1, 2])}

run_experiment(model=net, experiment_title=experiment_title, search_space=search_space, total_budget=total_budget,
               max_budget_per_config=max_budget_per_config, dataset_name=dataset_name, train_size=train_size,
               dtset_train=d_train, dtset_test=d_test, nb_cross_validation=nb_cross_validation, valid_size=0.10)
