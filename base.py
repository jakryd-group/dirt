""" Machine Learning on Data Without Preliminary Cleaning """

from torch import tanh
from torch import sqrt
import torch.nn as nn



class AutoEncoder(nn.Module):
    """ autoencoder class
        for now only holds:
            X variable
            y variable
    """

    def __init__(self, input_shape=45, encoded_shape=3):
        """
        initializing autoencoder class
        """
        super().__init__()                      # this is Python3 way

        half = int(input_shape / 2)
        quarter = int(input_shape / 4)
        eight = int(input_shape / 8)

        self.input_layer = nn.Linear(input_shape, half)
        self.encode_1 = nn.Linear(half, quarter)
        self.encode_2 = nn.Linear(quarter, eight)
        self.encode_3 = nn.Linear(eight, encoded_shape)

        self.decode_1 = nn.Linear(encoded_shape, eight)
        self.decode_2 = nn.Linear(eight, quarter)
        self.decode_3 = nn.Linear(quarter, half)
        self.output_layer = nn.Linear(half, input_shape)

        self.input = None

    def forward(self, x):
        """
        initialize forward method
        """
        x = self.encode(x)
        x = self.decode(x)

        return x

    def print(self):
        """ prints variables """
        print("X:", self.X, "\ny:", self.y)

    def encode(self, x):
        """
        encoding function
        """
        x = tanh(self.input_layer(x))
        x = tanh(self.encode_1(x))
        x = tanh(self.encode_2(x))

        return tanh(self.encode_3(x))

    def decode(self, z):
        """
        decoding function
        """
        z = tanh(self.decode_1(z))
        z = tanh(self.decode_2(z))
        z = tanh(self.decode_3(z))

        return tanh(self.output_layer(z))

    def trainIters(self, mod, optim):
        """
        training iterations method
        """
        def train_step(x):
            optim.zero_grad()
            predict = mod(x)
            los = self.lossfunction()
            los = los(predict, self.input)
            los = sqrt(self.criterion(x,predict))
            los.backward()
            optim.step()
            return los.item()

        return train_step
