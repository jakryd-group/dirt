""" Machine Learning on Data Without Preliminary Cleaning """

import torch
import torch.nn as nn
import torch.utils.data as Data
import mdshare
import numpy as np
import torchvision.transforms as transforms

import os

os.system("clear")

# Data load
dataset = mdshare.fetch("alanine-dipeptide-3x250ns-heavy-atom-distances.npz")#, working_directory=None)
with np.load(dataset) as f:
    X = np.vstack([f[key] for key in sorted(f.keys())])

num_workers = 0
batch_size = 20

# Data loader for easy mini-batch return in training 
train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
test_loader  = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class AutoEncoder(nn.Module):
    """ autoencoder class
        for now only holds:
            X variable
            y variable
    """

    def __init__(self, X, y=None):
        """
        initializing autoencoder class
        """
        super(AutoEncoder, self).__init__()
        self.X = X
        self.y = y

    def forward(self, x):
        """
        initialize forward method
        """
        # TO DO: implement here forward method
        
    def print(self):
        """ prints variables """
        print("X:", self.X, "\ny:", self.y)


    def _create_network(self):
        """
        initialize autoencode network weights and biases
        """
        # TO DO: implement here network creation

    def _initialize_weights():
        """
        initialize weights
        """
        # TO DO: implement here weights initialization

    def _encode(self):
        """
        encoding function
        """
        # TO DO: implement here encode function

    def _decode(self):
        """
        decoding function
        """
        # TO DO: implement here decode function

    def _loss_optimize():
        """
        loss optimizing function
        """
        # TO DO: implement here loss optimizing function

    def _transform():
        """
        transformation data into latent space
        """
        # TO DO: implement here transformation function

    def _train():
        """
        training function
        """
        # TO DO: implement here training function

    def _fit():
        """
        feed function
        """
        # TO DO: implement here fit function

    def _predict():
        """
        predict function
        """
        # TO DO: implement here predict function

    def lossfunction():
        """
        implement here lossfunction (output)
        """
        # TO DO: implement here loss function


#3 PRZECIĄZENIA forward
#implementacji lossfunction (wynik)
#jak wygląda metoda backpropagacji, liczenie błędów, odświeżanie parametrów
#iplemenetacja warstw 100 -> 10 -> 2 -> 10 -> 100

autoencoder = AutoEncoder(X)

'''
print(autoencoder)
print(type(autoencoder))
print(dir(autoencoder))
print(autoencoder.print())
'''

lis=[]
count = 0

lis.append((X[:], X[0]))
for i in np.asarray(lis[0][0]):
    print(i)
    count+=1
    if count == 50:
        break
