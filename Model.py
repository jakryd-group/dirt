""" Machine Learning on Data Without Preliminary Cleaning """

import torch

class Autoencoder(torch.nn.Module):
    """ autoencoder class
        for now only holds:
            X variable
            y variable
    """

    def __init__(self, input_shape):
        """
        initializing autoencoder class
        """
        super().__init__()                      # this is Python3 way
        
        self.input_shape = input_shape

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(True))

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, input_shape),
            torch.nn.Sigmoid())

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X
