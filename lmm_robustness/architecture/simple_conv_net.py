"""
https://github.com/berndprach/SimpleConvNet
"""
from functools import partial

from torch import nn

Conv = partial(nn.Conv2d, kernel_size=(3, 3), padding=(1, 1), bias=False)
Linear = partial(nn.Linear, bias=False)


def get_model() -> nn.Sequential:
    return nn.Sequential(
        bn_convolution(3, 64),

        bn_convolution(64, 64),
        bn_convolution_with_pooling(64, 128),

        bn_convolution(128, 128),
        bn_convolution_with_pooling(128, 256),

        bn_convolution(256, 256),
        bn_convolution_with_pooling(256, 512),

        bn_convolution(512, 512),

        nn.MaxPool2d(4),
        nn.Flatten(),
        Linear(512, 10)
    )


def bn_convolution(c_in: int, c_out: int):
    return nn.Sequential(
        Conv(c_in, c_out),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


def bn_convolution_with_pooling(c_in: int, c_out: int):
    return nn.Sequential(
        Conv(c_in, c_out),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )
