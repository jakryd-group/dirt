import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import numpy as np
from torch import chunk
from torch import randn
from torch import max
import io
import imageio
from PIL import Image


transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="torch_datasets/", train=True, transform=transform)

train_set, test_set = random_split(train_dataset, (48000, 12000))

BATCH_SIZE = 1000
BATCH_TEST_SIZE = 1000
EPOCHS = 50

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(
    test_set, batch_size=BATCH_TEST_SIZE, shuffle=False)


writer = SummaryWriter(
    'runs/%s; %d; %d' \
    % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S dn"), BATCH_SIZE, EPOCHS))


ite = iter(train_loader)

def add_noise(image, noise=0.2, normalize=True):
    """
    dodajemy szum do obrazów w celu uczenia
    """
    noise = randn(image.size()) * noise
    noisy_img = image + noise
    if normalize:
        noisy_img = noisy_img / max(noisy_img)
    return noisy_img


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf)

    # Closing the figure prevents it from being displayed directly
    plt.close(figure)

    buf.seek(0)
    png = Image.open(buf)
    png.load()

    image = Image.new("RGB", png.size, (255, 255, 255))
    image.paste(png, mask=png.split()[3]) # 3 is the alpha channel

    return image


def image_to_tensor(image):
    transf = transforms.Compose([transforms.ToTensor()])
    return transf(image)


def image_grid():
    """ Return a 5x5 grid of the MNIST images as a matplotlib figure. """
    # Create a figure to contain the plot.
    fig = plt.figure()#figsize=(10,10))

    images = next(ite)
    
    img = chunk(images[0], 200)

    raw = torchvision.utils.make_grid(img[0])
    raw = raw.numpy()
    raw = np.clip(raw, 0, 1)
    raw = np.transpose(raw, (1, 2, 0))

    img = torchvision.utils.make_grid(img[0])
    img = add_noise(img)
    img = img.numpy()
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))

    ax = fig.add_subplot(2, 1, 1)
    ax.set_title('Raw')
    ax.set_axis_off()
    plt.imshow(raw, cmap=plt.cm.binary)

    ax = fig.add_subplot(2, 1, 2)
    ax.set_title('Noise')
    ax.set_axis_off()
    plt.imshow(img, cmap=plt.cm.binary)

    return fig


"""for i in range(9):
    images = next(ite)
    #py.subplot(330 + 1 + i)
    
    img = chunk(images[0], 200)
    img = torchvision.utils.make_grid(img[0])
    
    raw = img.numpy()
    raw = np.transpose(raw, (1, 2, 0))

    img = add_noise(img)
    img = img.numpy()
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(raw)
    ax[1].imshow(img)



    plt.show()
"""

figure = image_grid()
ima = plot_to_image(figure)
#plt.imshow(ima)
#plt.show()
writer.add_image("test", image_to_tensor(ima), 0)
