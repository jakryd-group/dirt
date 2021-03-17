"""
próba wykorzystania klasy AutoEncoder poprzez import
"""
import os
import datetime
import torch
import torch.utils.data as Data
import mdshare
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter

from base import AutoEncoder


# Data load
DATA = mdshare.fetch("alanine-dipeptide-3x250ns-heavy-atom-distances.npz")
with np.load(DATA) as f:
    dataset = np.vstack([f[key] for key in sorted(f.keys())])

x_train, x_test = train_test_split(dataset, train_size=0.7)


#BATCH_SIZE = len(x_train) # pełen zbiór
BATCH_SIZE = 10000
#BATCH_SIZE = 45
EPOCHS = 1000
#EPOCHS = 10

writer = SummaryWriter(
    'runs/%s; %d; %d' \
    % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
       BATCH_SIZE, EPOCHS)
    )

def matplotlib_imshow(img):
    """ using plt to plot img in tensorboard """
    npimg = img.cpu().detach().numpy()
    npimg = npimg.transpose(1, 2, 0)

    plt.imshow(npimg.astype('uint8'))
    plt.close()


os.system("clear")
# pylint: disable=E1101




# Data loader for easy mini-batch return in training
train_loader = Data.DataLoader(
    dataset=x_train,
    batch_size=BATCH_SIZE,
    shuffle=True)
test_loader = Data.DataLoader(
    dataset=x_test,
    batch_size=BATCH_SIZE,
    shuffle=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(input_shape=45).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)
criterion = torch.nn.MSELoss()

# train = model.trainIters(model, optimizer)
losses = []

# Set model into train mode
model.train()

# Display info about train
print("### STARTING TRAINING ###")

last_loss = 0.0
batch_loss = 0.0
idx = 0

# start training
for epoch in range(EPOCHS):
    try:
        last_loss = batch_loss.item()
    except:
        last_loss = batch_loss

    batch_loss = 0.0


    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)
        batch_loss += loss
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        idx = batch_idx + idx
        writer.add_scalar('train/Batch loss', loss, idx)


    writer.add_scalar('train/Epoch loss', batch_loss, epoch)

    print('epoch: {}\t'.format(epoch + 1) \
        + 'avg loss: {:.10f}\t'.format(batch_loss / BATCH_SIZE) \
        + 'epoch loss: {:.5f}\t'.format(batch_loss))

    #print("last: %s\tbatch: %s" % (last_loss, batch_loss.item()))
    if epoch > 10 and last_loss < batch_loss.item():
        break



# Set model into evaluation mode
model.eval()

# Display info about test
print("### STARTING TESTING ###")


for batch_idx, data in enumerate(test_loader):
    batch_loss = 0.0
    if batch_idx == 0:
        writer.add_graph(model, data)
    data = data.to(device)
    output = model(data)

    img_input_grid = torchvision.utils.make_grid(data)
    matplotlib_imshow(img_input_grid)
    writer.add_image('input_data', img_input_grid, batch_idx)
    writer.add_embedding(data, tag='input_data', global_step=batch_idx)

    img_output_grid = torchvision.utils.make_grid(output)
    matplotlib_imshow(img_output_grid)
    writer.add_image('output_data', img_output_grid, batch_idx)
    writer.add_embedding(output, tag='output_data', global_step=batch_idx)

    enc = model.encode(data)
    img_encode_grid = torchvision.utils.make_grid(enc)
    matplotlib_imshow(img_encode_grid)
    writer.add_image('encoded_data', img_encode_grid, batch_idx)
    writer.add_embedding(enc, tag='encoded_data', global_step=batch_idx)

    loss = criterion(output, data)
    batch_loss += loss
    writer.add_scalar('test/Batch loss', loss, batch_idx)

    #if batch_idx == 9:
    #    break


    print('avg loss: {:.5f}\t'.format(batch_loss / BATCH_SIZE) \
        + 'test loss: {:.5f}\t'.format(batch_loss))
