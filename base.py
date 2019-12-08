""" Machine Learning on Data Without Preliminary Cleaning """

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import mdshare
import numpy as np

os.system("clear")
# pylint: disable=E1101

# Data load
DATA = mdshare.fetch("alanine-dipeptide-3x250ns-heavy-atom-distances.npz")#, working_directory=None)
with np.load(DATA) as f:
    dataset = np.vstack([f[key] for key in sorted(f.keys())])

BATCH_SIZE = 1000
BATCH_PART = 20
EPOCHS = 25

# Data loader for easy mini-batch return in training
train_loader = Data.DataLoader(dataset=dataset[::BATCH_PART], batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=dataset[::BATCH_PART], batch_size=BATCH_SIZE, shuffle=True)

class AutoEncoder(nn.Module):
    """ autoencoder class
        for now only holds:
            X variable
            y variable
    """

    #def __init__(self, x, y=None):
    def __init__(self):
        """
        initializing autoencoder class
        """
        super(AutoEncoder, self).__init__()

        #self.X = torch.from_numpy(x).type(torch.FloatTensor)
        #self.y = y

        self.layer1 = nn.Linear(45, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer2a = nn.Linear(10, 5)
        self.layer2b = nn.Linear(5, 2)

        self.layer3b = nn.Linear(2, 5)
        self.layer3a = nn.Linear(5, 10)
        self.layer3 = nn.Linear(10, 20)
        self.layer4 = nn.Linear(20, 45)

        self.input = None

    def forward(self, x):
        """
        initialize forward method
        """
        self.input = x

        x = self.encode(x)
        x = self.decode(x)

        return x

    def print(self):
        """ prints variables """
        print("X:", self.X, "\ny:", self.y)

    def _create_network(self):
        """
        initialize autoencode network weights and biases
        """
        # TO DO: implement here network creation

    def _initialize_weights(self):
        """
        initialize weights
        """
        # TO DO: implement here weights initialization

    def encode(self, x):
        """
        encoding function
        """

        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        h3 = F.relu(self.layer2a(h2))

        return self.layer2b(h3)

    def decode(self, z):
        """
        decoding function
        """

        h3 = F.relu(self.layer3b(z))
        h2 = F.relu(self.layer3a(h3))
        h1 = F.relu(self.layer3(h2))

        return torch.relu(self.layer4(h1))

    def _loss_optimize(self):
        """
        loss optimizing function
        """
        # TO DO: implement here loss optimizing function

    def _transform(self):
        """
        transformation data into latent space
        """
        # TO DO: implement here transformation function

    def _fit(self):
        """
        feed function
        """
        # TO DO: implement here fit function

    def _predict(self):
        """
        predict function
        """
        # TO DO: implement here predict function

    def lossfunction(self):
        """
        implement here lossfunction (output)
        """
        return nn.MSELoss(reduction='sum')

    #def train(self):
    #    """
    #    implement here train function
    #    """

    def trainIters(self, mod, optim):
        """
        training iterations method
        """
        def train_step(x):
            mod.train()
            predict = model(x)

            los = self.lossfunction()
            los = loss(self.input, predict)
            los.backward()

            optim.step()
            optim.zero_grad()

            return loss.item()

        return train_step


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train = model.trainIters(model, optimizer)
losses = []

for epoch in range(1, EPOCHS + 1):
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        batch = model(data)
        loss = train(batch)
        losses.append(loss)
        print('loss {:.24f}'.format(loss))
    print(epoch)

print(model.state_dict())
