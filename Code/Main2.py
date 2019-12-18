"""
    @file:              Main.py
    @Author:            Alexandre Ayotte
    @Creation Date:     30/09/2019
    @Last modification: 30/09/2019
    @Description:       Main file of the prototype program
"""

import DataManager as Dm
from HPtuner import HPtuner, ContinuousDomain, DiscreteDomain
import Model
import time
import numpy as np


# path = "C:/Users/Alexandre/Desktop/MAT523/AutoML-MAT523/Code/data/glass/glass.csv"
# x_train, t_train = Dm.load_csv(path, label_col=1, test_split=0.0)
# x_train, t_train, x_test, t_test = Dm.load_breast_cancer_dataset()


# print(x_train, t_train)
d_train, d_test = Dm.load_mnist()
input_size = np.array([28, 28, 3])
# ------------------------------------------------------------------------------------------
#                                       CNN VANILLA
# ------------------------------------------------------------------------------------------


# conv_layer = np.array([[16, 5, 1], [24, 5, 1], [24, 3, 1], [24, 3, 1], [32, 3, 0], [32, 3, 0]])
# pool_list = np.array([[0, 0, 0], [1, 2, 2], [0, 0, 0], [1, 2, 2], [0, 0, 0], [1, 2, 2]])
conv_layer = np.array([[16, 5, 1], [24, 5, 1]])
pool_list = np.array([[0, 0, 0], [1, 2, 2]])
fc_layer = np.array([256, 128, 64])

net = Model.CnnVanilla(10, conv_layer, pool_list, fc_layer, b_size=25, num_epoch=40, input_dim=input_size,
                       lr=0.0005, alpha=0.000, drop_rate=0.3, activation="relu", num_stop_epoch=5)
print(net)

net.set_hyperparameters({"activation": "swish"})

# ------------------------------------------------------------------------------------------
#                                           RESNET
# ------------------------------------------------------------------------------------------

# conv = np.array([16, 3, 1])
# res_config = np.array([[3, 3], [3, 3], [3, 3]])
# pool1 = np.array([0, 0, 0])  # No pooling layer after the first convolution
# pool2 = np.array([4, 1, 1])  # Average pooling after the first convolution layer.
# fc_nodes = None

"""
net = Model.ResNet(num_classes=10, conv_config=conv, res_config=res_config, pool1=pool1, pool2=pool2,
                   fc_config=fc_nodes, activation='relu', version=2, input_dim=input_size, lr=0.021457992,
                   alpha=0.001149, eps=0.1, drop_rate=0.0, b_size=100, num_epoch=10, num_stop_epoch=20,
                   lr_decay_rate=10, num_lr_decay=4, valid_size=0.05, tol=0.005)
"""

# net = Model.SimpleResNet(num_classes=10, num_res=3, activation='relu', version=1, input_dim=input_size, lr=0.021457992,
#                          alpha=0.001149, eps=0.1, mixup=[1, 0, 0], b_size=100, num_epoch=200, num_stop_epoch=20,
#                          lr_decay_rate=10, num_lr_decay=4, valid_size=0.05, tol=0.005, save_path="checkpoint.pth")

# print(net)


test = 'gaussian_process'

tune = HPtuner(net, test)

tune.set_search_space({'lr': ContinuousDomain(-6, -1, log_scaled=True),
                       'alpha': ContinuousDomain(-10,  -1, log_scaled=True),
                       'eps': DiscreteDomain([1e-8, 0.1, 1.0]),
                       'b_size': DiscreteDomain(np.arange(50, 360, 10).tolist())})

print(tune.search_space.space)
results = tune.tune(dtset=d_train, n_evals=50, valid_size=0.1, acquisition_function='MPI')
# results = tune.tune(dtset=d_train, n_evals=50, valid_size=0.1)

results.save_all_results('PLAIN_NET', "MNIST", 50000)


"""
net.fit(dtset=d_train, verbose=True, gpu=True)

score = net.score(dtset=d_test)

print("Test accuracy: {:.2f}".format(100 * score))
"""
