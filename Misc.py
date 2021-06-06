# -*- coding: utf-8 -*-
""" This module contains usefull functions """

import io
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.cm import binary
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import randn
from torch import max as tmax

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

def image_to_tensor(image):
    """ Converts image to pytorch tensor and returns it. """
    transf = transforms.Compose([transforms.ToTensor()])
    return transf(image)

#------------------------------------------------------------------------------

def create_plot_grid(*arg, names=['raw', 'noise', 'denoised']):
    """ Takes raw and dirty images and """
    count = len(arg)
    
    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout()
    

    for idx, img in enumerate(arg):
        tmp = img.numpy()

        norm = Normalize(vmin=0., vmax=1.)
        
        ax = fig.add_subplot(1, count, idx + 1)
        if names is not None:
            ax.set_title(names[idx])
        ax.set_axis_off()
        #plt.subplots_adjust(left=0.05, bottom=0.01, right=0.95, top=0.99, wspace=0.01, hspace=0.01)
        plt.imshow(tmp, cmap=plt.cm.gray)#, norm=norm)
    
    return fig

#------------------------------------------------------------------------------

def add_noise(image, noise, normalize):
    """
    Add noise to image
    """
    noise = randn(image.size()) * noise
    noisy_img = image + noise
    if normalize:
        noisy_img = noisy_img / tmax(noisy_img)
    return noisy_img

#------------------------------------------------------------------------------

def summary_grid(dirty, cleaned, noise=None):
    """
    Create summary image showing comparision between
    raw dataset and clened data
    noise input data can be also added
    """

