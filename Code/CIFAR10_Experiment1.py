"""
    @file:              Main.py
    @Author:            Alexandre Ayotte
    @Creation Date:     30/09/2019
    @Last modification: 27/11/2019
    @Description:       Experiment file on the CIFAR10 Dataset using residual network.
"""

import os
import DataManager as Dm
from HPtuner import HPtuner, ContinuousDomain, DiscreteDomain
import Model as Model
import numpy as np


# Generate training data
d_train, d_test = Dm.load_cifar10()
input_size = np.array([32, 32, 3])
results_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_path = os.path.join(results_path, '')
experiment_name = 'CF10Exp1'

# ------------------------------------------------------------------------------------------
#                                           RESNET
# ------------------------------------------------------------------------------------------

conv = np.array([16, 3, 1])
res_config = np.array([[3, 3], [3, 3], [3, 3]])
pool1 = np.array([0, 0, 0])  # No pooling layer after the first convolution
pool2 = np.array([4, 1, 1])  # Average pooling after the first convolution layer.
fc_nodes = None

net = Model.SimpleResNet(num_classes=10, num_res=3, activation='relu', version=1, input_dim=input_size, lr=0.021457992,
                         alpha=0.001149, eps=0.1, b_size=100, num_epoch=200, num_stop_epoch=20, lr_decay_rate=10,
                         num_lr_decay=4, valid_size=0.05, tol=0.004, save_path="checkpoint.pth")

print(net)

# ------------------------------------------------------------------------------------------
#                                       RANDOM SEARCH
# ------------------------------------------------------------------------------------------

test = 'random_search'

tune = HPtuner(net, test)

tune.set_search_space({'lr': ContinuousDomain(-7, -1, log_scaled=True),
                       'alpha': ContinuousDomain(-10,  -1, log_scaled=True),
                       'eps': DiscreteDomain([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]),
                       'b_size': DiscreteDomain(np.arange(50, 360, 10).tolist()),
                       'num_res': DiscreteDomain([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                       'lr_decay_rate': DiscreteDomain(np.arange(2, 40, 1).tolist()),
                       'activation': DiscreteDomain(['elu', 'relu', 'swish', 'mish']),
                       'version': DiscreteDomain([1, 2])})

print(tune.search_space.space)
results = tune.tune(dtset=d_train, n_evals=50, valid_size=0.1)

net.fit(dtset=d_train, verbose=True, gpu=True)

score = net.score(dtset=d_test)

results.save_all_results(results_path, experiment_name, "CIFAR10", 50000, score)

# ------------------------------------------------------------------------------------------
#                                           TPE
# ------------------------------------------------------------------------------------------

test = 'tpe'

tune = HPtuner(net, test)

tune.set_search_space({'lr': ContinuousDomain(-7, -1, log_scaled=True),
                       'alpha': ContinuousDomain(-10,  -1, log_scaled=True),
                       'eps': DiscreteDomain([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]),
                       'b_size': DiscreteDomain(np.arange(50, 360, 10).tolist()),
                       'num_res': DiscreteDomain([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                       'lr_decay_rate': DiscreteDomain(np.arange(2, 40, 1).tolist()),
                       'activation': DiscreteDomain(['elu', 'relu', 'swish', 'mish']),
                       'version': DiscreteDomain([1, 2])})

results = tune.tune(dtset=d_train, n_evals=50, valid_size=0.1)

net.fit(dtset=d_train, verbose=True, gpu=True)

score = net.score(dtset=d_test)

results.save_all_results(results_path, experiment_name, "CIFAR10", 50000, score)


# ------------------------------------------------------------------------------------------
#                                   GAUSSIAN PROCESS (MPI)
# ------------------------------------------------------------------------------------------

test = 'gaussian_process'

tune = HPtuner(net, test)

tune.set_search_space({'lr': ContinuousDomain(-7, -1, log_scaled=True),
                       'alpha': ContinuousDomain(-10,  -1, log_scaled=True),
                       'eps': DiscreteDomain([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]),
                       'b_size': DiscreteDomain(np.arange(50, 360, 10).tolist()),
                       'num_res': DiscreteDomain([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                       'lr_decay_rate': DiscreteDomain(np.arange(2, 40, 1).tolist()),
                       'activation': DiscreteDomain(['elu', 'relu', 'swish', 'mish']),
                       'version': DiscreteDomain([1, 2])})

results = tune.tune(dtset=d_train, n_evals=50, valid_size=0.1, acquisition_function='MPI')

net.fit(dtset=d_train, verbose=True, gpu=True)

score = net.score(dtset=d_test)

results.save_all_results(results_path, experiment_name, "CIFAR10", 50000, score)

# ------------------------------------------------------------------------------------------
#                                   GAUSSIAN PROCESS (EI)
# ------------------------------------------------------------------------------------------

test = 'gaussian_process'

tune = HPtuner(net, test)

tune.set_search_space({'lr': ContinuousDomain(-7, -1, log_scaled=True),
                       'alpha': ContinuousDomain(-10,  -1, log_scaled=True),
                       'eps': DiscreteDomain([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]),
                       'b_size': DiscreteDomain(np.arange(50, 360, 10).tolist()),
                       'num_res': DiscreteDomain([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                       'lr_decay_rate': DiscreteDomain(np.arange(2, 40, 1).tolist()),
                       'activation': DiscreteDomain(['elu', 'relu', 'swish', 'mish']),
                       'version': DiscreteDomain([1, 2])})

results = tune.tune(dtset=d_train, n_evals=50, valid_size=0.1, acquisition_function='EI')

net.fit(dtset=d_train, verbose=True, gpu=True)

score = net.score(dtset=d_test)

results.save_all_results(results_path, experiment_name, "CIFAR10", 50000, score)
