"""
"""
import torch
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np

from Model import Autoencoder as AE

import argparse

# Temporary fix to download the MNIST dataset
# https://github.com/pytorch/vision/issues/1938
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

#------------------------------------------------------------------------------

def train(model, data_train, optim, loss_fn, n_epochs):
  loss_list_result = []

  for epoch in range(n_epochs):
    loss = 0
    for inputs, _ in data_train:
      inputs = inputs.view(-1, 784)
      noise_img_train = add_noise(inputs)
      optim.zero_grad()
      outputs = model(noise_img_train)
      train_loss = loss_fn(outputs, inputs)
      train_loss.backward()
      optim.step()
      loss += train_loss.item()

    loss = loss / len(data_train)
    loss_list_result.append(loss)
      
    print('Epoch {} of {}, loss={:.3}'.format(epoch+1, args.n_epochs, loss))
  return loss_list_result

#------------------------------------------------------------------------------

def add_noise(image, noise=0.2, normalize=True):
  noise = torch.randn(image.size()) * noise
  noisy_img = image + noise
  if normalize:
    noisy_img = noisy_img / torch.max(noisy_img)
  return noisy_img

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(usage='python %(prog)s [options]',
                                 description='dirt: MNIST denoising using AE')
parser.add_argument('--n_epochs', type=int, required=False, default=100,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, required=False, default=128,
                    help='batch size')
parser.add_argument('--batch_size_test', type=int, required=False, default=10,
                    help='batch size for testing')
parser.add_argument('--noise', type=float, required=False, default=0.2,
                    help='noise level')
parser.add_argument('--normalize_img', type=bool, required=False, default=True,
                    help='normalize image with noise')
parser.add_argument('--learning_rate', type=float, required=False,
                    default=0.001, help='learning rate')
parser.add_argument('--verbose', type=bool, required=False, default=True,
                    help='verbose mode')
args = parser.parse_args()

#------------------------------------------------------------------------------

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root="datasets/", train=True,
                                           transform=transform, download=True)
train_set, test_set = random_split(train_dataset, (48000, 12000))
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=args.batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=args.batch_size_test,
                                          shuffle=False)

#------------------------------------------------------------------------------

device = None
if torch.cuda.is_available():
  device = 'cuda'
  if args.verbose: print('training on CUDA')
else:
  device = 'cpu'
  if args.verbose: print('training on CPU')

model = AE(input_shape=784).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.MSELoss()
if args.verbose:
  print('training started')
loss_list = train(model, train_loader, optimizer, criterion, args.n_epochs)
if args.verbose:
  print('training ended')

#------------------------------------------------------------------------------

# Check if AE works on a random test data
rnd = np.random.randint(len(test_set))
img = add_noise(test_set[rnd][0].view(1, 784), noise=args.noise,
                normalize=args.normalize_img)
out = model(img)
if args.normalize_img:
  out = out / out.max().item()
out = out.view(28, 28)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img.reshape(28, 28), cmap=plt.cm.Greys_r)
ax[1].imshow(out.detach(), cmap=plt.cm.Greys_r)
plt.xticks([])
plt.yticks([])
plt.show()
