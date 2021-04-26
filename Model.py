""" Machine Learning on Data Without Preliminary Cleaning """

from torch import sigmoid
from torch import sqrt
from torch import tanh
from torch import relu
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import ConvTranspose2d
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn.functional import batch_norm
from torch.nn.functional import relu


class AutoEncoder(Module):
    """ autoencoder class with linear layers"""

    def __init__(self, input_shape=45, encoded_shape=3):
        """ initializing class """
        super().__init__()                      # this is Python3 way

        half = int(input_shape / 2)
        quarter = int(input_shape / 4)
        eight = int(input_shape / 8)

        self.input_layer = Linear(input_shape, half)
        self.encode_1 = Linear(half, quarter)
        self.encode_2 = Linear(quarter, eight)
        self.encode_3 = Linear(eight, encoded_shape)

        self.decode_1 = Linear(encoded_shape, eight)
        self.decode_2 = Linear(eight, quarter)
        self.decode_3 = Linear(quarter, half)
        self.output_layer = Linear(half, input_shape)


    def forward(self, x):
        """ initialize forward method """
        x = self.encode(x)
        x = self.decode(x)
        return x


    def encode(self, x):
        """ encoding function """
        x = tanh(self.input_layer(x))
        x = tanh(self.encode_1(x))
        x = tanh(self.encode_2(x))
        return tanh(self.encode_3(x))


    def decode(self, z):
        """ decoding function """
        z = tanh(self.decode_1(z))
        z = tanh(self.decode_2(z))
        z = tanh(self.decode_3(z))
        return tanh(self.output_layer(z))

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


    def forward(self, x):
        """ initialize forward method """
        x = self.encode(x)
        x = self.decode(x)
        return x


    def encode(self, x):
        """ encoding function """
        x = relu(self.input_layer(x))
        return sigmoid(self.encode_1(x))


    def decode(self, z):
        """ decoding function """
        z = relu(self.decode_1(z))
        return relu(self.output_layer(z))

#------------------------------------------------------------------------------

class AutoencoderCNN(Module):
    """ autoencoder class with cnn layers"""

    def __init__(self):
        """ initializing class """
        super().__init__()                      # this is Python3 way

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


    def forward(self, X):
        """ defining model forward method """
        X = self.encode(X)
        X = self.decode(X)
        return X


    def encode(self, X):
        """ encoding function """
        X = relu(self.encode_1(X))
        X = self.bne1(X)
        X = relu(self.encode_2(X))
        X = self.bne2(X)
        X = relu(self.encode_3(X))
        X = self.bne3(X)
        return X


    def decode(self, X):
        """ decoding function """
        X = relu(self.decode_1(X))
        X = self.bnd(X)
        X = relu(self.decode_2(X))
        X = sigmoid(self.decode_3(X))
        return X
