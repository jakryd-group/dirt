# -*- coding: utf-8 -*-
""" Using Convolutional Neutral Network to denoise MNIST images """

import datetime
import numpy as np
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
from six.moves import urllib
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tensorboardX import SummaryWriter

from Arguments import parser
from Misc import add_noise
from Misc import create_plot_grid
from Misc import image_to_tensor
from Misc import plot_to_image
from Misc import summary_grid
from Model import AutoEncoderCNN

#------------------------------------------------------------------------------

def train(model, data_train, optim, loss_fn, epochs, noise, norm):
    """ train function """
    loss_list_result = []
    for epoch in range(epochs):
        loss = 0
        for inputs, _ in data_train:
            # add noise
            noise_img = add_noise(inputs, noise=noise, normalize=norm)
            # reset gradients
            optim.zero_grad()
            # move model (ie. net) to appropiate device
            model.to(device)
            # compute reconstructions
            outputs = model(noise_img.to(device))
            # compute a training reconstruction loss
            train_loss = loss_fn(outputs.to(device), inputs.to(device))
            # compute accumulated gradients
            train_loss.backward()
            # update parameters based on current gradients
            optim.step()
            #add the batch training loss to epoch loss
            loss += train_loss.item()
        # compute the epoch training loss
        loss = loss / len(data_train)
        # and add it to results list
        loss_list_result.append(loss)

        # print info every epoch
        if args.verbose and not args.suppress:
            print('Epoch {} of {}, loss={:.3}'.format(epoch+1, epochs, loss))
        else:
            # print info every 10th epoch
            if epoch%10 == 0 and not args.suppress:
                print('Epoch {} of {}, loss={:.3}'.format(epoch+1, epochs, loss))

        # store training progress in tensorboard if requested
        if args.tensorboard:
            writer.add_scalar('train/Epoch loss', loss, epoch)
            writer.flush()

    return loss_list_result

#------------------------------------------------------------------------------

def test(model, test_data, noise, normalize):
    """ test function """
    count = 0

    for img, _ in test_data:
        noise_img = add_noise(img, noise=noise, normalize=normalize)

        out = model(noise_img.to(device))
        out = out/out.max().item()
        out = out.detach().cpu()

        img = torch.chunk(img, args.batch_size_test)
        noise_img = torch.chunk(noise_img, args.batch_size_test)
        out = torch.chunk(out, args.batch_size_test)

        # store training progress in tensorboard if requested
        if args.tensorboard:
            writer.add_image(
                'test_model',
                image_to_tensor(
                    plot_to_image(
                        create_plot_grid(
                            img[0].view(28, 28),
                            noise_img[0].view(28, 28),
                            np.clip(out[0].view(28, 28), 0., 1.),
                            names=['raw', 'noise %s' % noise, 'denoised']))),
                count)

        # display summary
        if count == 0:
            dims = (1400, 600)
            summary = Image.new('L', (dims[0], 5 * dims[1]), 255)
            i = 0
            while i < 5:
                tmp_img = plot_to_image(
                            create_plot_grid(
                                img[i].view(28, 28),
                                noise_img[i].view(28, 28),
                                out[i].view(28, 28),
                                names=None))
                summary.paste(tmp_img, (0, i*dims[1]))
                i += 1

            plt.imshow(summary, cmap=plt.cm.gray)
            plt.show()

        count = count + 1

#------------------------------------------------------------------------------

# Temporary fix to download the MNIST dataset
# https://github.com/pytorch/vision/issues/1938

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

#------------------------------------------------------------------------------

# set default parameters
args = parser(desc='Denoise MNIST dataset using convolutional activation layer',
              n_epochs=1,
              batch_size=1000,
              batch_test_size=1000)

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

# store training progress if needed
if args.tensorboard:
    writer = SummaryWriter(
        'runs/%s; %d; %d' \
        % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S dn cnn"),
           args.batch_size, args.n_epochs))

#------------------------------------------------------------------------------

# find out if CUDA capable GPU is present
device = None
if torch.cuda.is_available():
    device = 'cuda'
    if args.verbose and not args.suppress:
        print('training on CUDA')
else:
    device = 'cpu'
    if args.verbose and not args.suppress:
        print('training on CPU')

#------------------------------------------------------------------------------

# declare model, loss function
AEmodel = AutoEncoderCNN().to(device)
optimizer = torch.optim.Adam(
    AEmodel.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay)
criterion = torch.nn.MSELoss()

#------------------------------------------------------------------------------

# training
if args.verbose and not args.suppress:
    print('training started')

loss_list = train(AEmodel,
                  train_loader,
                  optimizer,
                  criterion,
                  args.n_epochs,
                  args.noise,
                  args.normalize_img)

if args.verbose and not args.suppress:
    print('training ended')

#------------------------------------------------------------------------------

# disable AEmodel train mode
AEmodel.eval()

# testing
if args.verbose and not args.suppress:
    print('testing started')
test(AEmodel, test_loader, args.noise, args.normalize_img)
if args.verbose and not args.suppress:
    print('testing ended')
