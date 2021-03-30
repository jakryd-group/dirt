""" Machine Learning on Data Without Preliminary Cleaning """

from torch import sigmoid
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


from base import AutoEncoder

class Autoencoder(Module):
    """ autoencoder class """

    def __init__(self, input_shape):
        """
        initializing autoencoder class
        """
        super().__init__()                      # this is Python3 way


        """self.input_layer  = Conv2d(1, 64, 3, 1)
        self.output_layer = Conv2d(64, 1, 3, 1)

        self.encode_1 = Conv2d(64, 32, 3, 1)
        self.encode_2 = Conv2d(32, 28, 3, 1)

        self.decode_1 = Conv2d(28, 32, 3, 1)
        self.decode_2 = Conv2d(32, 64, 3, 1)

        self.bn1 = BatchNorm2d(64)
        self.bn2 = BatchNorm2d(32)
        self.bn3 = BatchNorm2d(28)
        """
        """self.encode_1 = Conv2d( 1, 64, 3, 1, 1)
        self.encode_2 = Conv2d(64, 32, 3, 1, 1)
        self.encode_3 = Conv2d(32, 16, 3, 1, 1)
        
        self.pool = MaxPool2d(8, 8)
        self.bn1 = BatchNorm2d(64)
        self.bn2 = BatchNorm2d(32)
        self.bn3 = BatchNorm2d(16)

        self.decode_1 = ConvTranspose2d(16, 32, 3, 1, 1)
        self.decode_2 = ConvTranspose2d(32, 64, 3, 1, 1)
        self.decode_3 = ConvTranspose2d(64, 1,  28, 1, 1)
        """
        self.encode_1 = Conv2d( 1, 64, kernel_size=3)
        self.encode_2 = Conv2d(64, 32, kernel_size=3)
        self.encode_3 = Conv2d(32, 16, kernel_size=3)
        
        self.pool = MaxPool2d(2, stride=1)
        self.bn1 = BatchNorm2d(64)
        self.bn2 = BatchNorm2d(32)
        self.bn3 = BatchNorm2d(16)

        self.decode_1 = ConvTranspose2d(16, 32, kernel_size=3)
        self.decode_2 = ConvTranspose2d(32, 64, kernel_size=3)
        self.decode_3 = Conv2d(64, 1, kernel_size=3, padding=2)

    def forward(self, X):
        """
        defining model forward method
        """
        X = self.encode(X)
        X = self.decode(X)
        return X

    def encode(self, X):
        """ encoding function
        X = self.bn1(relu(self.input_layer(X)))
        X = self.bn2(relu(self.encode_1(X)))
        return self.bn3(relu(self.encode_2(X)))
        """
        X = relu(self.encode_1(X))
        X = self.bn1(X)

        X = relu(self.encode_2(X))
        X = self.bn2(X)

        X = relu(self.encode_3(X))
        X = self.bn3(X)

        return X
        
    def decode(self, X):
        """ decoding function 
        X = self.bn3(relu(self.decode_1(X)))
        X = relu(self.decode_2(X))
        return Sigmoid(self.output_layer(X))
        """
        X = relu(self.decode_1(X))
        X = self.bn2(X)
        X = relu(self.decode_2(X))

        X = sigmoid(self.decode_3(X))

        return X
