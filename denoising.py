"""
Próba wykorzystania klasy AutEncoder do odszumienia MNIST
"""
import datetime
from os import write
import random
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import torch
from torch.utils.data import random_split
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from base import AutoEncoder
import copy
import numpy as np
from PIL import Image

def add_noise(image):
    """
    dodajemy szum do obrazów w celu uczenia
    """
    noise = torch.randn(image.size()) * 0.2
    noisy_img = image + noise
    noisy_img = noisy_img/torch.max(noisy_img)
    return noisy_img


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="torch_datasets/", train=True, transform=transform)

train_set, test_set = random_split(train_dataset, (48000, 12000))


BATCH_SIZE = 1000
BATCH_TEST_SIZE = 1000
EPOCHS = 100

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_TEST_SIZE, shuffle=False)


writer = SummaryWriter(
    'runs/%s; %d; %d' \
    % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S dn"), BATCH_SIZE, EPOCHS))


#plt.imshow(train_set[0][0].reshape((28,28)), cmap='Greys')
#plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AEmodel = AutoEncoder(input_shape=784, encoded_shape=10).to(device)
optimizer = torch.optim.Adam(AEmodel.parameters(), lr=0.001, weight_decay=1e-7)
criterion = torch.nn.MSELoss()


def train(model, data_train, optim, loss_fn, epochs):
    """
    train function
    """
    loss_list_result = []
    for epoch in range(epochs):
        loss = 0
        for inputs, _ in data_train:
            #reshape inputs
            inputs = inputs.view(-1, 784)
            #add some noise
            noise_img_train = add_noise(inputs)
            #reser gradients
            optim.zero_grad()
            #compute reconstructions
            outputs = model(noise_img_train)
            #compute a training reconstruction loss
            train_loss = loss_fn(outputs, inputs)
            #compute accumulated gradients
            train_loss.backward()
            #update parameters based on current gradients
            optim.step()
            #add the batch training loss to epoch loss
            loss += train_loss.item()
        #compute the epoch training loss
        loss = loss/len(data_train)
        loss_list_result.append(loss)
        #print info every 10th epoch
        if epoch%10 == 0:
            print('Epoch {} of {}, loss={:.3}'.format(epoch+1, epochs, loss))
        writer.add_scalar('train/Epoch loss', loss, epoch)
    return loss_list_result


loss_list = train(AEmodel, train_loader, optimizer, criterion, EPOCHS)


counter = 0

for img, _ in test_loader:
    noise = add_noise(img)
    noise = noise.view(-1, 784)
    out = AEmodel(noise)
    out = out/out.max().item()
    out = out.detach().cpu()

    img = img.view(-1, 784)
    img = torch.chunk(img, BATCH_TEST_SIZE)
    noise = torch.chunk(noise, BATCH_TEST_SIZE)
    out = torch.chunk(out, BATCH_TEST_SIZE)

    #torch.cat((img.view(28, 28), noise.view(28, 28), out.view(28, 28)), 1),
    grid = torchvision.utils.make_grid(
        torch.cat((img[0].view(28, 28), noise[0].view(28, 28), out[0].view(28, 28)), 1),
        nrow=3,
        padding=100)
    writer.add_image('test_model', grid, counter)
    counter = counter + 1
