from Model import AutoencoderCNN
from Misc import plot_to_image
from Misc import image_to_tensor
from Misc import create_plot_grid
from torchvision import transforms
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tensorboardX import SummaryWriter
import datetime
from Arguments import args
import numpy as np



def get_data(path):
    images = []
    for image in os.listdir(path):
        image = Image.open(os.path.join(path, image))
        image.load()
        image.thumbnail((256, 256))

       
        images.append(image) 

    return images

#------------------------------------------------------------------------------

def train(model, data_train, optim, loss_fn, n_epochs):
    loss_list_result = []

    for epoch in range(n_epochs):
        loss = 0
        for inputs, _ in data_train:
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

# prepare the dataset and the dataloader
class ImageData(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]
        data = np.asarray(data).astype(np.uint8).reshape((256, 256, 3))
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            labels = self.y[i]
            labels = np.asarray(labels).astype(np.uint8).reshape((256, 256, 3))
            labels = self.transforms(labels)
            return (data, labels)
        else:
            return data


#------------------------------------------------------------------------------

EPOCHS = 1
BATCH_SIZE = 2

#------------------------------------------------------------------------------

transform = transforms.Compose([transforms.ToTensor()])

train_images = get_data('./Dirty-Documents/train')
train_clean = get_data('./Dirty-Documents/train_cleaned')
test_images = get_data('./Dirty-Documents/test')

train_data = ImageData(train_images, train_clean, transform)
test_data = ImageData(test_images, None, transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

#------------------------------------------------------------------------------

writer = SummaryWriter(
    'runs/%s; %d; %d' \
    % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S docs cnn"), BATCH_SIZE, EPOCHS))

#------------------------------------------------------------------------------

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

AEmodel = AutoencoderCNN().to(device)
optimizer = torch.optim.Adam(AEmodel.parameters(), lr=0.001, weight_decay=1e-7)
criterion = torch.nn.MSELoss()

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


for img, _ in test_loader:
    img = img.view(-1, 784)
    noise_img = add_noise(img, noise=ratios)

    if counter == 0:
        writer.add_graph(AEmodel, noise_img.to(device))

    out = AEmodel(noise_img.to(device))
    out = out/out.max().item()
    out = out.detach().cpu()

    img = torch.chunk(img, BATCH_TEST_SIZE)
    noise_img = torch.chunk(noise_img, BATCH_TEST_SIZE)
    out = torch.chunk(out, BATCH_TEST_SIZE)

    #plt.imshow(noise_img[0].view(28, 28).detach().cpu().numpy())
    #plt.show()

    #grid = make_grid(
    #    torch.cat((img[0].view(28, 28), noise_img[0].view(28, 28), out[0].view(28, 28)), 1),
    #    nrow=3,
    #    padding=100)
    #writer.add_image('test_model_%s' % (ratios), grid, counter)
    writer.add_image(
        'test_model',
        image_to_tensor(
            plot_to_image(
                create_plot_grid(
                    img[0].view(28, 28),
                    noise_img[0].view(28, 28),
                    out[0].view(28, 28),
                    names=['raw', 'noise %s' % (ratios), 'denoised']
                )
            )
        ),
        counter)

    counter = counter + 1