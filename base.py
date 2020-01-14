""" Machine Learning on Data Without Preliminary Cleaning """

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import mdshare
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/tmp_1')

def matplotlib_imshow(img):
    """ using plt to plot img in tensorboard """
    img = img.mean(dim=0)
    npimg = img.cpu().detach().numpy()
    plt.imshow(npimg, cmap="Greys")

os.system("clear")
# pylint: disable=E1101

# Data load
DATA = mdshare.fetch("alanine-dipeptide-3x250ns-heavy-atom-distances.npz")
with np.load(DATA) as f:
    dataset = np.vstack([f[key] for key in sorted(f.keys())])

x_train, x_test = train_test_split(dataset, train_size=0.7)

BATCH_SIZE = 1000
BATCH_PART = 20
EPOCHS = 15

# Data loader for easy mini-batch return in training
train_loader = Data.DataLoader(
    dataset=x_train,
    batch_size=BATCH_SIZE,
    shuffle=True
    )
test_loader = Data.DataLoader(
    dataset=x_test,
    batch_size=BATCH_SIZE,
    shuffle=True
    )


class AutoEncoder(nn.Module):
    """ autoencoder class
        for now only holds:
            X variable
            y variable
    """

    # def __init__(self, x, y=None):
    def __init__(self):
        """
        initializing autoencoder class
        """
        super(AutoEncoder, self).__init__()

        # self.X = torch.from_numpy(x).type(torch.FloatTensor)
        # self.y = y

        self.input_layer = nn.Linear(45, 20)
        self.encode_1 = nn.Linear(20, 10)
        self.encode_2 = nn.Linear(10, 5)
        self.encode_3 = nn.Linear(5, 2)

        self.decode_1 = nn.Linear(2, 5)
        self.decode_2 = nn.Linear(5, 10)
        self.decode_3 = nn.Linear(10, 20)
        self.output_layer = nn.Linear(20, 45)

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

        x = F.relu(self.input_layer(x))
        x = F.relu(self.encode_1(x))
        x = F.relu(self.encode_2(x))

        return F.relu(self.encode_3(x))

    def decode(self, z):
        """
        decoding function
        """

        z = F.relu(self.decode_1(z))
        z = F.relu(self.decode_2(z))
        z = F.relu(self.decode_3(z))

        return F.relu(self.output_layer(z))

    def trainIters(self, mod, optim):
        """
        training iterations method
        """
        def train_step(x):
            mod.train()
            predict = mod(x)

            optim.zero_grad()

            los = self.lossfunction()
            los = los(self.input, predict)

            los.backward()

            optim.step()

            return los.item()

        return train_step


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = torch.nn.MSELoss()

# train = model.trainIters(model, optimizer)
losses = []

# Set model into train mode
model.train()

# train
for epoch in range(EPOCHS):
    batch_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        # print(f'Epoch: {epoch} batch_idx: {batch_idx}')
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)
        batch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch: {}\t'.format(epoch + 1) \
        + 'avg loss: {:.5f}\t'.format(batch_loss / 525) \
        + 'loss: {:.5f}\t'.format(batch_loss))



# Set model into evaluation mode
model.eval()
for batch_idx, data in enumerate(test_loader):
    data = data.to(device)
    output = model(data)
    #print(f'Data {data} \nOutput {output}')

    #dataiter = iter(data)
    #input_img = dataiter.next()
    img_input_grid = torchvision.utils.make_grid(data)
    matplotlib_imshow(img_input_grid)
    writer.add_image('input_data', img_input_grid, batch_idx)
    print(data[0:10])

    img_output_grid = torchvision.utils.make_grid(output)
    matplotlib_imshow(img_output_grid)
    writer.add_image('output_data', img_output_grid, batch_idx)
    print(output[0:10])
