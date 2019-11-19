"""
    @file:              Model.py
    @Author:            Nicolas Raymond
                        Alexandre Ayotte
    @Creation Date:     30/09/2019
    @Last modification: 17/11/2019

    @Reference: 1)  K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

    @Description:       This program generates models of different types to use for our classification problems.
                        This is a version that contain our deprecated classes and method.
"""

import numpy as np
import Model as Mod
import torch
import Module as Module


class CnnVanilla(Mod.Cnn):
    def __init__(self, num_classes, conv_layer, pool_list, fc_nodes, activation='relu', input_dim=None, lr=0.001,
                 alpha=0.0, eps=1e-8, drop_rate=0.5, b_size=15, num_epoch=10, valid_size=0.10, tol=0.005,
                 num_stop_epoch=10, lr_decay_rate=5, num_lr_decay=3):

        """
        Class that generate a convolutional neural network using the Pytorch library

        :param num_classes: Number of class
        :param conv_layer: A Cx3 numpy matrix where each row represent the parameters of a 2D convolutional layer.
                           [i, 0]: Number of output channels of the ith layer
                           [i, 1]: Square convolution dimension of the ith layer
                           [i, 2]: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param pool_list: An Cx3 numpy matrix where each row represent the parameters of a 2D pooling layer.
                          [i, 0]: Pooling layer type: 0: No pooling, 1: Max pooling, 2: Average pooling
                          [i, 1]: Pooling kernel height
                          [i, 2]: Pooling kernel width
        :param fc_nodes: A numpy array where each elements represent the number of nodes of a fully connected layer
        :param input_dim: Image input dimensions [height, width, deep]
        :param activation: Activation function (default: relu)
        :param lr: The initial learning rate used with the Adam Optimizer
        :param alpha: L2 penalty (regularization term) parameter as float (default: 0.0)
        :param eps: Adam optimizer hyper-parameters used to improve numerical stability (default: 1e-8)
        :param drop_rate: Dropout rate of each node of all fully connected layer (default: 0.5
        :param b_size: Batch size as integer (default: 15)
        :param num_epoch: Number of epoch to do during the training (default: 10)
        :param valid_size: Portion of the data that will be used for validation.
        :param tol: Minimum difference between two epoch validation accuracy to consider that there is an improvement.
        :param num_stop_epoch: Number of consecutive epoch with no improvement on the validation accuracy
                               before early stopping
        :param lr_decay_rate: Rate of the learning rate decay when the optimizer does not seem to converge
        :param num_lr_decay: Number of learning rate decay step we do before stop training when the optimizer does not
                             seem to converge.
        """

        Mod.Cnn.__init__(self, num_classes, activation=activation, lr=lr, alpha=alpha, eps=eps, drop_rate=drop_rate,
                         b_size=b_size, num_epoch=num_epoch, valid_size=valid_size, tol=tol,
                         num_stop_epoch=num_stop_epoch, lr_decay_rate=lr_decay_rate, num_lr_decay=num_lr_decay)

        # We need a special type of list to ensure that torch detect every layer and node of the neural net
        self.cnn_layer = torch.nn.ModuleList()
        self.fc_layer = torch.nn.ModuleList()
        self.num_flat_features = 0
        self.out_layer = None
        self.pool = pool_list

        # Default image dimension. Height: 28, width: 28 and deep: 1 (MNIST)
        if input_dim is None:
            input_dim = np.array([28, 28, 1])

        # We build the model
        self.build_layer(conv_layer, pool_list, fc_nodes, input_dim)

    def build_layer(self, conv_layer, pool_list, fc_nodes, input_dim):

        """
        Create the model architecture

        :param conv_layer: A Cx3 numpy matrix where each row represent the parameters of a 2D convolutional layer.
                           [i, 0]: Number of output channels of the ith layer
                           [i, 1]: Square convolution dimension of the ith layer
                           [i, 2]: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param pool_list: An Cx3 numpy matrix where each row represent the parameters of a 2D pooling layer.
                          [i, 0]: Pooling layer type: 0: No pooling, 1: Max pooling, 2: Average pooling
                          [i, 1]: Pooling kernel height
                          [i, 2]: Pooling kernel width
        :param fc_nodes: A numpy array where each elements represent the number of nodes of a fully connected layer
        :param input_dim: Image input dimensions [height, width, deep]
        :return:
        """

        # ------------------------------------------------------------------------------------------
        #                                   CONVOLUTIONAL PART
        # ------------------------------------------------------------------------------------------
        # First convolutional layer
        self.cnn_layer.append(torch.nn.Conv2d(input_dim[2], conv_layer[0, 0], conv_layer[0, 1],
                                              padding=self.pad_size(conv_layer[0, 1], conv_layer[0, 2])
                                              ))
        # Activation function
        self.cnn_layer.append(self.get_activation_function())

        # Pooling layer
        if pool_list[0, 0] != 0:
            self.cnn_layer.append(self.build_pooling_layer(pool_list[0]))

        # We need to compute the input size of the fully connected layer
        size = self.conv_out_size(input_dim[0:2], conv_layer[0, 1], conv_layer[0, 2], pool_list[0])

        # All others convolutional layers
        for it in range(1, len(conv_layer)):
            self.cnn_layer.append(torch.nn.Conv2d(conv_layer[it - 1, 0], conv_layer[it, 0], conv_layer[it, 1],
                                                  padding=self.pad_size(conv_layer[it, 1], conv_layer[it, 2])
                                                  ))
            self.cnn_layer.append(self.get_activation_function())

            if pool_list[it, 0] != 0:
                self.cnn_layer.append(self.build_pooling_layer(pool_list[it]))

            # Update the output size
            size = self.conv_out_size(size, conv_layer[it, 1], conv_layer[it, 2], pool_list[it])

        # ------------------------------------------------------------------------------------------
        #                                   FULLY CONNECTED PART
        # ------------------------------------------------------------------------------------------
        # Compute the fully connected input layer size
        self.num_flat_features = size[0] * size[1] * conv_layer[-1, 0]

        # First fully connected layer
        self.fc_layer.append(torch.nn.Linear(self.num_flat_features, fc_nodes[0]))
        self.fc_layer.append(self.get_activation_function())
        self.fc_layer.append(self.drop)

        # All others hidden layers
        for it in range(1, len(fc_nodes)):
            self.fc_layer.append(torch.nn.Linear(fc_nodes[it - 1], fc_nodes[it]))
            self.fc_layer.append(self.get_activation_function())
            self.fc_layer.append(self.drop)

        # Output layer
        self.out_layer = torch.nn.Linear(fc_nodes[-1], self.classes)

    def forward(self, x):

        """
        Define the forward pass of the neural network

        :param x: Input tensor of size BxD where B is the Batch size and D is the features dimension
        :return: Output tensor of size num_class x 1.
        """

        for i, l in enumerate(self.cnn_layer):
            x = self.cnn_layer[i](x) + l(x)

        x = x.view(-1, self.num_flat_features)

        for i, l in enumerate(self.fc_layer):
            x = self.fc_layer[i](x) + l(x)

        x = self.soft(self.out_layer(x))
        return x


class ResNet(Mod.Cnn):
    def __init__(self, num_classes, conv, res_config, pool1, pool2, fc_nodes, activation='relu', input_dim=None,
                 lr=0.001, alpha=0.0, eps=1e-8, drop_rate=0.0, b_size=15, num_epoch=10, valid_size=0.10, tol=0.005,
                 num_stop_epoch=10, lr_decay_rate=5, num_lr_decay=3):

        """
        Class that generate a ResNet neural network inpired by the model from the paper "Deep Residual Learning for
        Image Recogniton" (Ref 1).

        :param num_classes: Number of classes
        :param conv: A tuple that represent the parameters of the first convolutional layer.
                     [0]: Number of output channels (features maps)
                     [1]: Kernel size: (Example: 3.  For a 3x3 kernel)
                     [2]: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param res_config: A Cx2 numpy matrix where each row represent the parameters of a sub-sampling level.
                           [i, 0]: Number of residual modules
                           [i, 2]: Kernel size of the convolutional layers
        :param pool1: A tuple that represent the parameters of the pooling layer that came after the first conv layer
        :param pool2: A tuple that represent the parameters of the last pooling layer before the fully-connected layers
                      [0]: Pooling layer type: 0: No pooling, 1: Max pooling, 2: Average pooling
                      [1]: Pooling kernel height
                      [2]: Pooling kernel width
        :param fc_nodes: A numpy array where each elements represent the number of nodes of a fully connected layer
        :param input_dim: Image input dimensions [height, width, deep]
        :param activation: Activation function (default: relu)
        :param lr: The initial learning rate used with the Adam Optimizer
        :param alpha: L2 penalty (regularization term) parameter as float (default: 0.0)
        :param eps: Adam optimizer hyper-parameters used to improve numerical stability (default: 1e-8)
        :param drop_rate: Dropout rate of each node of all fully connected layer (default: 0.5
        :param b_size: Batch size as integer (default: 15)
        :param num_epoch: Number of epoch to do during the training (default: 10)
        :param valid_size: Portion of the data that will be used for validation.
        :param tol: Minimum difference between two epoch validation accuracy to consider that there is an improvement.
        :param num_stop_epoch: Number of consecutive epoch with no improvement on the validation accuracy
                               before early stopping
        :param lr_decay_rate: Rate of the learning rate decay when the optimizer does not seem to converge
        :param num_lr_decay: Number of learning rate decay step we do before stop training when the optimizer does not
                             seem to converge.
        """

        Mod.Cnn.__init__(self, num_classes, activation=activation, lr=lr, alpha=alpha, eps=eps, drop_rate=drop_rate,
                         b_size=b_size, num_epoch=num_epoch, valid_size=valid_size, tol=tol,
                         num_stop_epoch=num_stop_epoch, lr_decay_rate=lr_decay_rate, num_lr_decay=num_lr_decay)

        # We need a special type of list to ensure that torch detect every layer and node of the neural net
        self.cnn_layer = torch.nn.ModuleList()
        self.fc_layer = torch.nn.ModuleList()
        self.num_flat_features = 0
        self.out_layer = None

        # Default image dimension. Height: 32, width: 32 and deep: 3 (CIFAR10)
        if input_dim is None:
            input_dim = np.array([32, 32, 3])

        self.build_layer(conv, res_config, pool1, pool2, fc_nodes, input_dim)

    def build_layer(self, conv, res_config, pool1, pool2, fc_nodes, input_dim):

        """
        Create the model architecture

        :param conv: A tuple that represent the parameters of the first convolutional layer.
                     [0]: Number of output channels (features maps)
                     [1]: Kernel size: (Example: 3.  For a 3x3 kernel)
                     [2]: Convolution type: (0: Valid (no zero padding), 1: Same (zero padding added))
        :param res_config:A Cx2 numpy matrix where each row represent the parameters of a sub-sampling level.
                          [i, 0]: Number of residual modules
                          [i, 1]: Kernel size of the convolutional layers
        :param pool1: A tuple that represent the parameters of the pooling layer that came after the first conv layer
        :param pool2: A tuple that represent the parameters of the last pooling layer before the fully-connected layers
                      [0]: Pooling layer type: 0: No pooling, 1: Max pooling, 2: Average pooling
                      [1]: Pooling kernel height
                      [2]: Pooling kernel width
        :param fc_nodes: A numpy array where each elements represent the number of nodes of a fully connected layer
        :param input_dim: Image input dimensions [height, width, deep]
        """

        # ------------------------------------------------------------------------------------------
        #                                   CONVOLUTIONAL PART
        # ------------------------------------------------------------------------------------------
        # First convolutional layer
        self.cnn_layer.append(torch.nn.Conv2d(input_dim[2], conv[0], conv[1], padding=self.pad_size(conv[1], conv[2])))
        self.cnn_layer.append(torch.nn.BatchNorm2d(conv[0]))
        self.cnn_layer.append(self.get_activation_function())

        if pool1[0] != 0:
            self.cnn_layer.append(self.build_pooling_layer(pool1))

        # We need to compute the input size of the fully connected layer
        size = self.conv_out_size(input_dim[0:2], conv[1], conv[2], pool1)

        # ------------------------------------------------------------------------------------------
        #                                      RESIDUAL PART
        # ------------------------------------------------------------------------------------------

        f_in = conv[0]

        for it in range(len(res_config)):
            self.cnn_layer.append(Module.ResModule(f_in, res_config[it, 1],
                                                   self.activation, twice=(it != 0), subsample=(it != 0)))

            # Update
            if it > 0:
                f_in *= 2
                size = size / 2

            for _ in range(res_config[it, 0] - 1):
                self.cnn_layer.append(Module.ResModule(f_in, res_config[it, 1],
                                                       self.activation, twice=False, subsample=False))

        if pool2[0] != 0:
            self.cnn_layer.append(self.build_pooling_layer(pool2))

        # We need to compute the input size of the fully connected layer
        size = self.conv_out_size(size, res_config[-1, 1], 2, pool2)

        # ------------------------------------------------------------------------------------------
        #                                   FULLY CONNECTED PART
        # ------------------------------------------------------------------------------------------
        # Compute the fully connected input layer size
        self.num_flat_features = size[0] * size[1] * f_in

        if fc_nodes is None:
            num_last_nodes = self.num_flat_features
        else:
            # First fully connected layer
            self.fc_layer.append(torch.nn.Linear(self.num_flat_features, fc_nodes[0]))
            self.fc_layer.append(self.get_activation_function())
            self.fc_layer.append(self.drop)

            # All others hidden layers
            for it in range(1, len(fc_nodes)):
                self.fc_layer.append(torch.nn.Linear(fc_nodes[it - 1], fc_nodes[it]))
                self.fc_layer.append(self.get_activation_function())
                self.fc_layer.append(self.drop)

            num_last_nodes = fc_nodes[-1]

        # Output layer
        self.out_layer = torch.nn.Linear(num_last_nodes, self.classes)

    def forward(self, x):

        """
        Define the forward pass of the neural network

        :param x: Input tensor of size BxD where B is the Batch size and D is the features dimension
        :return: Output tensor of size num_class x 1.
        """

        for i, l in enumerate(self.cnn_layer):
            x = self.cnn_layer[i](x) + l(x)

        x = x.view(-1, self.num_flat_features)

        for i, l in enumerate(self.fc_layer):
            x = self.fc_layer[i](x) + l(x)

        x = self.soft(self.out_layer(x))
        return x