""" Machine Learning on Data Without Preliminary Cleaning """
import os
import datetime
import torch
import torch.nn as nn
import torch.utils.data as Data
import mdshare
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter

#BATCH_SIZE = len(x_train) # pełen zbiór
BATCH_SIZE = 1000
#BATCH_SIZE = 45
#EPOCHS = 50
EPOCHS = 10

writer = SummaryWriter(
    'runs/%s; %d; %d' \
    % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
       BATCH_SIZE, EPOCHS)
    )

def matplotlib_imshow(img):
    """ using plt to plot img in tensorboard """
    npimg = img.cpu().detach().numpy()
    npimg = npimg.transpose(1, 2, 0)

    plt.imshow(npimg.astype('uint8'))
    plt.close()


os.system("clear")
# pylint: disable=E1101

# Data load
DATA = mdshare.fetch("alanine-dipeptide-3x250ns-heavy-atom-distances.npz")
with np.load(DATA) as f:
    dataset = np.vstack([f[key] for key in sorted(f.keys())])


x_train, x_test = train_test_split(dataset, train_size=0.7)


# Data loader for easy mini-batch return in training
train_loader = Data.DataLoader(
    dataset=x_train,
    batch_size=BATCH_SIZE,
    shuffle=True)
test_loader = Data.DataLoader(
    dataset=x_test,
    batch_size=BATCH_SIZE,
    shuffle=True)


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
        #super(AutoEncoder, self).__init__()    # this is the old way
        super().__init__()                      # this is Python3 way

        self.input_layer = nn.Linear(45, 20)
        self.encode_1 = nn.Linear(20, 10)
        self.encode_2 = nn.Linear(10, 5)
        self.encode_3 = nn.Linear(5, 2)

        self.decode_1 = nn.Linear(2, 5)
        self.decode_2 = nn.Linear(5, 10)
        self.decode_3 = nn.Linear(10, 20)
        self.output_layer = nn.Linear(20, 45)

        self.running_loss = 0.0

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
        x = torch.tanh(self.input_layer(x))
        x = torch.tanh(self.encode_1(x))
        x = torch.tanh(self.encode_2(x))

        return torch.tanh(self.encode_3(x))

    def decode(self, z):
        """
        decoding function
        """
        z = torch.tanh(self.decode_1(z))
        z = torch.tanh(self.decode_2(z))
        z = torch.tanh(self.decode_3(z))

        return torch.tanh(self.output_layer(z))

    def trainIters(self, mod, optim):
        """
        training iterations method
        """
        def train_step(x):
            optim.zero_grad()
            predict = mod(x)
            los = self.lossfunction()
            los = los(predict, self.input)
            los.backward()
            optim.step()
            return los.item()

        return train_step


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)
criterion = torch.nn.MSELoss()

# train = model.trainIters(model, optimizer)
losses = []

# Set model into train mode
model.train()

# Display info about train
print("### STARTING TRAINING ###")

# start training
for epoch in range(EPOCHS):
    batch_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)
        batch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/Batch loss', loss, batch_idx)

    writer.add_scalar('train/Epoch loss', batch_loss, epoch)

    print('epoch: {}\t'.format(epoch + 1) \
        + 'avg loss: {:.10f}\t'.format(batch_loss / BATCH_SIZE) \
        + 'epoch loss: {:.5f}\t'.format(batch_loss))


# Set model into evaluation mode
model.eval()

# Display info about test
print("### STARTING TESTING ###")

batch_loss = 0.0
for batch_idx, data in enumerate(test_loader):
    if batch_idx == 0:
        writer.add_graph(model, data)
    data = data.to(device)
    output = model(data)

    img_input_grid = torchvision.utils.make_grid(data)
    matplotlib_imshow(img_input_grid)
    writer.add_image('input_data', img_input_grid, batch_idx)
    writer.add_embedding(data, tag='input_data', global_step=batch_idx)

    img_output_grid = torchvision.utils.make_grid(output)
    matplotlib_imshow(img_output_grid)
    writer.add_image('output_data', img_output_grid, batch_idx)
    writer.add_embedding(output, tag='output_data', global_step=batch_idx)

    enc = model.encode(data)
    img_encode_grid = torchvision.utils.make_grid(enc)
    matplotlib_imshow(img_encode_grid)
    writer.add_image('encoded_data', img_encode_grid, batch_idx)
    writer.add_embedding(enc, tag='encoded_data', global_step=batch_idx)

    loss = criterion(output, data)
    batch_loss += loss
    writer.add_scalar('test/Batch loss', loss, batch_idx)


print('avg loss: {:.5f}\t'.format(batch_loss / BATCH_SIZE) \
    + 'test loss: {:.5f}\t'.format(batch_loss))
