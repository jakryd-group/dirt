# -*- coding: utf-8 -*-
""" Machine Learning on Data Without Preliminary Cleaning """

from torch import sigmoid
from torch import relu
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import ConvTranspose2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Module

#------------------------------------------------------------------------------

class AutoEncoderRelu(Module):
    """ autoencoder class with linear layers"""

    def __init__(self):
        """ initializing class """
        super().__init__()

        self.input_layer = Linear(784, 256)
        self.encode_1 = Linear(256, 64)

        self.decode_1 = Linear(64, 256)
        self.output_layer = Linear(256, 784)


    def forward(self, value):
        """ initialize forward method """
        value = self.encode(value)
        value = self.decode(value)
        return value


    def encode(self, value):
        """ encoding function """
        value = relu(self.input_layer(value))
        return relu(self.encode_1(value))


    def decode(self, value):
        """ decoding function """
        value = relu(self.decode_1(value))
        return sigmoid(self.output_layer(value))

#------------------------------------------------------------------------------

class AutoEncoderCNN(Module):
    """ autoencoder class with cnn layers"""

    def __init__(self):
        """ initializing class """
        super().__init__()

        self.encode_1 = Conv2d( 1, 32, kernel_size=3)
        self.encode_2 = Conv2d(32, 16, kernel_size=3)
        self.encode_3 = Conv2d(16, 8, kernel_size=3)

        self.pool = MaxPool2d(2, stride=1)
        self.bne1 = BatchNorm2d(32)
        self.bne2 = BatchNorm2d(16)
        self.bne3 = BatchNorm2d(8)

        self.bnd = BatchNorm2d(16)

        self.decode_1 = ConvTranspose2d(8, 16, kernel_size=3)
        self.decode_2 = ConvTranspose2d(16, 32, kernel_size=3)
        self.decode_3 = Conv2d(32, 1, kernel_size=3, padding=2)


    def forward(self, value):
        """ defining model forward method """
        value = self.encode(value)
        value = self.decode(value)
        return value


    def encode(self, value):
        """ encoding function """
        value = relu(self.encode_1(value))
        value = self.bne1(value)
        value = relu(self.encode_2(value))
        value = self.bne2(value)
        value = relu(self.encode_3(value))
        value = self.bne3(value)
        return value


    def decode(self, value):
        """ decoding function """
        value = relu(self.decode_1(value))
        value = self.bnd(value)
        value = relu(self.decode_2(value))
        value = sigmoid(self.decode_3(value))
        return value
