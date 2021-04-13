"""
Próba wykorzystania klasy AutEncoder do odszumienia MNIST
"""
import datetime
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import torch
from torch.utils.data import random_split
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Model import AutoEncoderRelu
from Misc import plot_to_image
from Misc import image_to_tensor
from Misc import create_plot_grid
import numpy as np

def add_noise(image, noise=0.2, normalize=True):
    """
    dodajemy szum do obrazów w celu uczenia
    """
    noise = torch.randn(image.size()) * noise
    noisy_img = image + noise
    if normalize:
        noisy_img = noisy_img / torch.max(noisy_img)
    return noisy_img


transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="torch_datasets/", train=True, transform=transform)

train_set, test_set = random_split(train_dataset, (48000, 12000))


BATCH_SIZE = 1000
BATCH_TEST_SIZE = 1000
EPOCHS = 25
NOISE_RATIO = [0.0, 0.05, 0.2, 0.5, 1.0]

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_TEST_SIZE, shuffle=False)


writer = SummaryWriter(
    'runs/%s; %d; %d' \
    % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S dn relu"), BATCH_SIZE, EPOCHS))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AEmodel = AutoEncoderRelu(input_shape=784, encoded_shape=10).to(device)
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
            # reshape inputs
            inputs = inputs.view(-1, 784)
            inputs = np.clip(inputs, 0, 1)
            # add some noise
            noise_img_train = add_noise(inputs)
            # reset gradients
            optim.zero_grad()
            # move model (ie. net) to appropiate device
            model.to(device)
            # compute reconstructions
            outputs = model(noise_img_train.to(device))
            # compute a training reconstruction loss
            train_loss = loss_fn(outputs.to(device), inputs.to(device))
            # compute accumulated gradients
            train_loss.backward()
            # update parameters based on current gradients
            optim.step()
            #a dd the batch training loss to epoch loss
            loss += train_loss.item()
        # compute the epoch training loss
        loss = loss/len(data_train)
        # loss = loss/len(NOISE_RATIO)
        loss_list_result.append(loss)
        # print info every 10th epoch
        if epoch%10 == 0:
            print('Epoch {} of {}, loss={:.3}'.format(epoch+1, epochs, loss))
        writer.add_scalar('train/Epoch loss', loss, epoch)

        for name, weight in model.named_parameters():
            writer.add_histogram(name,weight, epoch)
            writer.add_histogram(f'{name}.grad',weight.grad, epoch)

        writer.flush()
    return loss_list_result


loss_list = train(AEmodel, train_loader, optimizer, criterion, EPOCHS)

counter = 0

for ratios in NOISE_RATIO:
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
