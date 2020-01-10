import DataManager as Dm
from HPtuner import HPtuner, ContinuousDomain, DiscreteDomain
import Model
import numpy as np

d_train, d_test = Dm.load_cifar10()
input_size = np.array([28, 28, 3])

net = Model.SimpleResNet(10, 3, activation='relu', version=1, input_dim=input_size, lr=0.01, alpha=4.53e-5, eps=1e-3,
                         mixup=[2, 2, 2], b_size=150, num_epoch=300, valid_size=0.10, tol=0.005, num_stop_epoch=40,
                         lr_decay_rate=15, num_lr_decay=3, save_path="Checkpoint.pth")

#################################################
# PreActResNet18
#################################################
conv = np.array([64, 3, 1])
res = np.array([[2, 3, 2], [2, 3, 2], [2, 3, 2], [2, 3, 2]])
pool1 = np.array([0, 0, 0])  # No pooling layer after the first convolution
pool2 = np.array([4, 1, 1])  # Adaptive average pooling after the last convolution layer.
fc_config = None  # No extra fully connected after the the convolutional part.

prenet = Model.ResNet(num_classes=10, conv_config=conv, res_config=res, pool1=pool1, pool2=pool2,
                      fc_config=fc_config, activation='relu', version=2, input_dim=input_size, lr=0.01,
                      alpha=4.53e-5, eps=1e-3, b_size=150, num_epoch=300, num_stop_epoch=40,
                      lr_decay_rate=15, num_lr_decay=4, valid_size=0.1, tol=0.005, save_path="Checkpoint.pth")

print(prenet)

prenet.fit(dtset=d_train, verbose=True)

prenet.restore()

score = prenet.score(dtset=d_test)

print(score)
