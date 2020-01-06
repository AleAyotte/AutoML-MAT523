import DataManager as Dm
from HPtuner import HPtuner, ContinuousDomain, DiscreteDomain
import Model
import numpy as np

d_train, d_test = Dm.load_cifar10()
input_size = np.array([28, 28, 3])

net = Model.SimpleResNet(10, 3, activation='relu', version=1, input_dim=input_size, lr=0.01, alpha=4.53e-5, eps=1e-3,
                         mixup=[0, 2, 2], b_size=150, num_epoch=30, valid_size=0.10, tol=0.005, num_stop_epoch=10,
                         lr_decay_rate=15, num_lr_decay=3, save_path="Checkpoint.pth")

net.fit(dtset=d_train, verbose=True)

score = net.score(dtset=d_test)

