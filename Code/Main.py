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

net = Model.SimpleResNet(num_classes=10, num_res=7, activation='elu', version=1, input_dim=input_size, lr=0.021457992,
                         alpha=0.001149, eps=0.1, b_size=100, num_epoch=200, num_stop_epoch=20, lr_decay_rate=25,
                         num_lr_decay=4, valid_size=0.05, tol=0.004, save_path="checkpoint.pth")

print(net)

net.fit(dtset=d_train, verbose=True, gpu=True)

score = net.score(dtset=d_test)

print(score)

