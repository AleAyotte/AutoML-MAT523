"""
    @file:              Module.py
    @Author:            Nicolas Raymond
                        Alexandre Ayotte
    @Creation Date:     30/09/2019
    @Last modification: 02/11/2019

    @Reference: 1)  K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
                2)  Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Swish:a self-gated activation function."
                    arXiv preprint arXiv:1710.05941 7 (2017)
                3)  Diganta Misra. "Mish: A self Regularized Non-Monotonic Neural Activation Function"
                    arXiv preprint arXiv:1908.08681v2 (2019)
                4) Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep
                   residual networks. CoRR, abs/1603.05027, 2016
    @Description:       This program generates modules used to construct the CNN model.
"""

import torch
import torch.nn.functional as F
import Model


class ResModuleV1(torch.nn.Module):
    def __init__(self, fmap_in, kernel, activation, bias=False, twice=False, subsample=False):

        """
        Create a residual block from the paper "Deep Residual Learning for Image Recogniton" (Ref 1)
        @Inspired by: https://github.com/a-martyn/resnet/blob/master/resnet.py

        :param fmap_in: Number of feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param activation: Activation function (default: relu)
        :param bias: If we want bias at convolutional layer
        :param twice: If we want twice more features at the output
        :param subsample: If we want to subsample the image.
        """

        torch.nn.Module.__init__(self)

        if activation == "relu":
            self.activation1 = torch.nn.ReLU()
        elif activation == "preLu":
            self.activation1 = torch.nn.PReLU()
        elif activation == "elu":
            self.activation1 = torch.nn.ELU()
        elif activation == "sigmoide":
            self.activation1 = torch.nn.Sigmoid()
        elif activation == "swish":
            self.activation1 = Swish()
        elif activation == "mish":
            self.activation1 = Mish()

        self.activation2 = self.activation1  # Do we need a deep copy?

        # Build layer
        fmap_out = 2*fmap_in if twice else fmap_in

        self.conv1 = torch.nn.Conv2d(fmap_in, fmap_out, kernel_size=kernel, stride=(2 if subsample else 1),
                                     padding=Model.Cnn.pad_size(kernel, 1), bias=bias)
        self.bn1 = torch.nn.BatchNorm2d(fmap_out)

        self.conv2 = torch.nn.Conv2d(fmap_out, fmap_out, kernel_size=kernel, stride=1,
                                     padding=Model.Cnn.pad_size(kernel, 1), bias=bias)
        self.bn2 = torch.nn.BatchNorm2d(fmap_out)

        # If subsample is True
        self.subsample = subsample
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=1, stride=2)

    def forward(self, x):

        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.subsample:
            avg_x = self.avg_pool(x)
            x = torch.cat((avg_x, torch.zeros_like(avg_x)), dim=1)

        out = self.activation2(out + x)

        return out


class ResModuleV2(torch.nn.Module):
    def __init__(self, fmap_in, kernel, activation, bias=False, twice=False, subsample=False):

        """
        Create a residual block from the paper "Identity Mappings in Deep Residual Networks" (Ref 1)
        @Inspired by: https://github.com/a-martyn/resnet/blob/master/resnet.py

        :param fmap_in: Number of feature maps
        :param kernel: Kernel size as integer (Example: 3.  For a 3x3 kernel)
        :param activation: Activation function (default: relu)
        :param bias: If we want bias at convolutional layer
        :param twice: If we want twice more features at the output
        :param subsample: If we want to subsample the image.
        """

        torch.nn.Module.__init__(self)

        if activation == "relu":
            self.activation1 = torch.nn.ReLU()
        elif activation == "preLu":
            self.activation1 = torch.nn.PReLU()
        elif activation == "elu":
            self.activation1 = torch.nn.ELU()
        elif activation == "sigmoide":
            self.activation1 = torch.nn.Sigmoid()
        elif activation == "swish":
            self.activation1 = Swish()
        elif activation == "mish":
            self.activation1 = Mish()

        self.activation2 = self.activation1  # Do we need a deep copy?

        # Build layer
        fmap_out = 2*fmap_in if twice else fmap_in

        self.bn1 = torch.nn.BatchNorm2d(fmap_in)

        self.conv1 = torch.nn.Conv2d(fmap_in, fmap_out, kernel_size=kernel, stride=(2 if subsample else 1),
                                     padding=Model.Cnn.pad_size(kernel, 1), bias=bias)

        self.bn2 = torch.nn.BatchNorm2d(fmap_out)

        self.conv2 = torch.nn.Conv2d(fmap_out, fmap_out, kernel_size=kernel, stride=1,
                                     padding=Model.Cnn.pad_size(kernel, 1), bias=bias)

        # If subsample is True
        self.subsample = subsample
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=1, stride=2)

    def forward(self, x):

        """
        Define the forward pass of the Residual layer

        :param x: Input tensor of the convolutional layer
        :return: Output tensor of the residual block
        """

        out = self.bn1(x)
        out = self.activation1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)

        if self.subsample:
            avg_x = self.avg_pool(x)
            x = torch.cat((avg_x, torch.zeros_like(avg_x)), dim=1)

        out = out + x

        return out


class Swish(torch.nn.Module):
    def __init__(self):

        """
        This the constructor of the swish activation function
        """

        torch.nn.Module.__init__(self)

    def forward(self, x):

        """
        Define the forward pass of the swish activation function
        swish(x) = x * sigmoid(x)

        :param x: Input tensor of size Bx... where B is the Batch size and ... correspond to the other dimension
        :return: Output tensor of the same sime as the input
        """
        return x * torch.sigmoid(x)


class Mish(torch.nn.Module):
    def __init__(self):

        """
        This the constructor of the mish activation function
        """

        torch.nn.Module.__init__(self)

    def forward(self, x):

        """
        Define the forward pass of the mish activation function
        mish(x) = x * tanh(softplus(x))

        :param x: Input tensor of size Bx... where B is the Batch size and ... correspond to the other dimension
        :return: Output tensor of the same sime as the input
        """
        return x * torch.tanh(F.softplus(x))
