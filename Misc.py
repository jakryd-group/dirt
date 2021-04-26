import io
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.cm import binary
import numpy as np
from PIL import Image
from torchvision import transforms


def plot_to_image(figure):
    """ Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call. """
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
    """ Converts image to pytorch tensor and returns it. """
    transf = transforms.Compose([transforms.ToTensor()])
    return transf(image)


def create_plot_grid(*arg, names=['raw', 'noise', 'denoised']):
    """ Takes raw and dirty images and """
    count = len(arg)
    
    fig = plt.figure(figsize=(14, 6))
    

    for idx, img in enumerate(arg):
        tmp = img.numpy()

        norm = Normalize(vmin=0., vmax=1.)
        
        ax = fig.add_subplot(1, count, idx + 1)
        ax.set_title(names[idx])
        ax.set_axis_off()
        plt.subplots_adjust(left=0.01, bottom=0.01, right=1, top=1, wspace=0.05, hspace=0.1)
        plt.imshow(tmp, cmap=plt.cm.gray)#, norm=norm)
    
    return fig
