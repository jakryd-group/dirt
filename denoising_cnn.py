""" Using Convolutional Neutral Network to denoise text images """

from torch import randn
from torch import max
from torch import cuda
from torch import chunk
from torch.nn import MSELoss
from torch.optim import Adam
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib

from Model import AutoencoderCNN as AE
from Arguments import args


# Temporary fix to download the MNIST dataset
# https://github.com/pytorch/vision/issues/1938

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

#------------------------------------------------------------------------------

def train(model, data_train, optim, loss_fn, n_epochs):
    loss_list_result = []

    for epoch in range(n_epochs):
        loss = 0
        for inputs, _ in data_train:
            #inputs = inputs.view(-1, 784)
            noise_img_train = add_noise(inputs)
            optim.zero_grad()
            model.to(device)
            outputs = model(noise_img_train.to(device))
            train_loss = loss_fn(outputs.to(device), inputs.to(device))
            train_loss.backward()
            optim.step()
            loss += train_loss.item()

        loss = loss / len(data_train)
        loss_list_result.append(loss)

        print('Epoch {} of {}, loss={:.3}'.format(epoch+1, args.n_epochs, loss))
    return loss_list_result

#------------------------------------------------------------------------------

def add_noise(image, noise=0.2, normalize=True):
    """
    adding noise to torch tensors
    this function is used for model train purposes
    """
    noise = randn(image.size()) * noise
    noisy_img = image + noise
    if normalize:
        noisy_img = noisy_img / max(noisy_img)
    return noisy_img

#------------------------------------------------------------------------------

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root="datasets/", train=True,
                                           transform=transform, download=True)
train_set, test_set = random_split(train_dataset, (48000, 12000))
train_loader = DataLoader(train_set,
                            batch_size=args.batch_size,
                            shuffle=True)
test_loader = DataLoader(test_set,
                            batch_size=args.batch_size_test,
                            shuffle=False)

#------------------------------------------------------------------------------

device = None
if cuda.is_available():
    device = 'cuda'
    if args.verbose:
        print('training on CUDA')
else:
    device = 'cpu'
    if args.verbose:
        print('training on CPU')

AEmodel = AE(input_shape=784).to(device)
optimizer = Adam(AEmodel.parameters(), lr=args.learning_rate)
criterion = MSELoss()
if args.verbose:
    print('training started')
loss_list = train(AEmodel, train_loader, optimizer, criterion, args.n_epochs)
if args.verbose:
    print('training ended')

#------------------------------------------------------------------------------

# Check if AE works on a random test data
rnd = np.random.randint(len(test_set))

# Disable AEmodel train mode
AEmodel.eval()


#img = add_noise(test_set[rnd][0].view(1, 784), noise=args.noise,
#    normalize=args.normalize_img)
"""img = add_noise(test_set[rnd][0], noise=args.noise,
    normalize=args.normalize_img)
out = AEmodel(img.to(device))
if args.normalize_img:
    out = out / out.max().item()
out = out.view(28, 28)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img.reshape(28, 28), cmap=plt.cm.Greys_r)
ax[1].imshow(out.detach().cpu(), cmap=plt.cm.Greys_r)
plt.xticks([])
plt.yticks([])
plt.show()
"""

for img, _ in test_loader:
    #img = img.view(-1, 784)
    noise_img = add_noise(img)

    out = AEmodel(noise_img.to(device))
    out = out/out.max().item()
    out = out.detach().cpu()

    img = chunk(img, args.batch_size_test)
    noise_img = chunk(noise_img, args.batch_size_test)
    out = chunk(out, args.batch_size_test)

    #plt.imshow(noise_img[0].view(28, 28).detach().cpu().numpy())
    #plt.show()

    #grid = make_grid(
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img[0].view(28, 28), cmap=plt.cm.Greys_r)
    ax[1].imshow(noise_img[0].view(28, 28), cmap=plt.cm.Greys_r)
    ax[2].imshow(out[0].view(28, 28), cmap=plt.cm.Greys_r)
    plt.xticks([])
    plt.yticks([])
    plt.show()
