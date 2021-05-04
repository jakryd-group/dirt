# -*- coding: utf-8 -*-
""" Using Convolutional Neutral Network to denoise text images """

import os
import datetime
import torch
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Arguments import parser
from Misc import plot_to_image
from Misc import image_to_tensor
from Misc import create_plot_grid
from Model import AutoEncoderCNN

#------------------------------------------------------------------------------

def get_data(path):
    """
    Gets images from provided as arg path folder
    """
    images = []
    dimensions = (540, 420)
    for image in os.listdir(path):
        image = Image.open(os.path.join(path, image))
        image.load()

        width, height = image.size
        w_padding = int((dimensions[0] - width) / 2)
        h_padding = int((dimensions[1] - height) / 2)

        tmp = Image.new('L', dimensions, 255)
        tmp.paste(image, (w_padding, h_padding))

        images.append(tmp)

    return images

#------------------------------------------------------------------------------

def train(model, data_train, optim, loss_fn, n_epochs):
    """
    train function
    """
    loss_list_result = []

    for epoch in range(n_epochs):
        loss = 0
        for dirt, clean in data_train:
            dirt = torch.autograd.Variable(dirt, requires_grad=True).cuda()
            clean = torch.autograd.Variable(clean, requires_grad=True).cuda()
            optim.zero_grad()
            model.to(device)
            outputs = model(dirt.to(device))
            train_loss = loss_fn(outputs.to(device), clean.to(device))
            train_loss.backward()
            optim.step()
            loss += train_loss.item()

        loss = loss / len(data_train)
        loss_list_result.append(loss)

        print('Epoch {} of {}, loss={:.3}'.format(epoch+1, args.n_epochs, loss))
        writer.add_scalar('train/Epoch loss', loss, epoch)
        writer.flush()
    return loss_list_result

#------------------------------------------------------------------------------

def test(model, testdata):
    """
    test function
    """
    count = 0

    for dirty in testdata:
        if count == 0:
            writer.add_graph(model, dirty.to(device))

        out = AEmodel(dirty.to(device))
        out = out/out.max().item()
        out = out.detach().cpu()

        dirty = torch.chunk(dirty, args.batch_size_test)
        out = torch.chunk(out, args.batch_size_test)

        writer.add_image(
            'test_model',
            image_to_tensor(
                plot_to_image(
                    create_plot_grid(
                        dirty[0].view(-1, *dirty[0].size()[2:]).permute(1, 2, 0),
                        out[0].view(-1, *out[0].size()[2:]).permute(1, 2, 0),
                        names=['dirty', 'denoised']
                    )
                )
            ),
            count)

        count = count + 1

#------------------------------------------------------------------------------

# prepare the dataset and the dataloader
class ImageData(Dataset):
    """
    Custom DataLoader class
    """
    def __init__(self, dirty, cleaned=None, trans=None):
        self.dirt = dirty
        self.clean = cleaned
        self.transforms = trans

    def __len__(self):
        return len(self.dirt)

    def __getitem__(self, i):
        data = self.dirt[i]
        data = np.asarray(data).astype(np.uint8)

        if self.transforms:
            data = self.transforms(data)

        if self.clean is not None:
            labels = self.clean[i]
            labels = np.asarray(labels).astype(np.uint8)
            labels = self.transforms(labels)
            return (data, labels)

        return data

#------------------------------------------------------------------------------

# set default parameters
args = parser(desc='Denoise MNIST dataset using convolutional activation layer',
              n_epochs=10,
              batch_size=8,
              batch_test_size=8)

#------------------------------------------------------------------------------

transform = transforms.Compose([transforms.ToTensor()])

train_images = get_data('./Dirty-Documents/train')
train_clean = get_data('./Dirty-Documents/train_cleaned')
test_images = get_data('./Dirty-Documents/test')

train_data = ImageData(train_images, train_clean, transform)
test_data = ImageData(test_images, None, transform)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True)

#------------------------------------------------------------------------------

writer = SummaryWriter(
    'runs/%s; %d; %d' \
    % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S docs cnn"),
       args.batch_size, args.n_epochs))

#------------------------------------------------------------------------------

# find out if CUDA capable GPU is present
device = None
if torch.cuda.is_available():
    device = 'cuda'
    if args.verbose:
        print('training on CUDA')
else:
    device = 'cpu'
    if args.verbose:
        print('training on CPU')

#------------------------------------------------------------------------------

# declare model, loss function
AEmodel = AutoEncoderCNN().to(device)
optimizer = torch.optim.Adam(
    AEmodel.parameters(),
    weight_decay=args.weight_decay,
    lr=args.learning_rate)
criterion = torch.nn.MSELoss()

#------------------------------------------------------------------------------

# training
if args.verbose:
    print('training started')
loss_list = train(AEmodel, train_loader, optimizer, criterion, args.n_epochs)
if args.verbose:
    print('training ended')

#------------------------------------------------------------------------------

# disable AEmodel train mode
AEmodel.eval()

# testing
if args.verbose and not args.suppress:
    print('testing started')
test(AEmodel, test_loader)
if args.verbose and not args.suppress:
    print('testing ended')
