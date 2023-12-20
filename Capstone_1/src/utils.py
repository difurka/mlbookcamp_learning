from time import time

import numpy as np

import os

from skimage.io import imread
from torch.utils.data import DataLoader
from skimage.transform import resize

import torch

import subprocess

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def get_datasets(batch_size):
    if not os.path.exists('./data'):
        os.mkdir('./data')
   
    runcmd("wget https://www.dropbox.com/s/8lqrloi0mxj2acu/PH2Dataset.rar", verbose = True)
    runcmd("unrar x PH2Dataset.rar ./data", verbose = True)

    os.remove('PH2Dataset.rar')
    images = []
    lesions = []

    
    root = './data/PH2Dataset'
    i = 0
    for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset images')):
        if root.endswith('_Dermoscopic_Image'):
            images.append(imread(os.path.join(root, files[0])))
        if root.endswith('_lesion'):
            lesions.append(imread(os.path.join(root, files[0])))


    size = (256, 256)
    X = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images]
    Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]

    import numpy as np
    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)
    print(f'Loaded {len(X)} images')


    """Let's divide our 200 pictures by 100/50/50
    for training, validation and test respectively
    """

    ix = np.random.choice(len(X), len(X), False)
    tr, val, ts = np.split(ix, [100, 150])

    """#### PyTorch DataLoader"""

    data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])),
                        batch_size=batch_size, shuffle=True)
    data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])),
                        batch_size=batch_size, shuffle=True)
    data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])),
                        batch_size=batch_size, shuffle=True)

    return data_tr, data_val, data_ts


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    print(outputs.shape, labels.shape)
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return thresholded  #


def bce_loss(y_real, y_pred):
    BCE = y_pred - y_real * y_pred +torch.log(1+torch.exp(-y_pred))
    return BCE.mean()


def score_model(model, metric, data, treshold=0.5):
    model.eval()  # testing mode
    scores = 0
    with torch.no_grad():
      for X_batch, Y_label in data:
          print('X_batch:', type(X_batch))
          Y_pred = torch.sigmoid(model(X_batch))
          Y_pred = torch.where(Y_pred > treshold, 1, 0)
          scores += metric(Y_pred, Y_label).mean().item()
    return scores/len(data)


def train(model, opt, loss_fn, epochs, lr, data_tr, data_val, metric=iou_pytorch):
    X_val, Y_val = next(iter(data_val))
    losses = {'train': [], 'valid': []}
    scores = []
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(data_tr))
    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))
        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:

            # set parameter gradients to zero

            opt.zero_grad()
            # forward
            Y_pred = model(X_batch)
            loss =  loss_fn(Y_batch, Y_pred)
            loss.backward() # backward-pass
            opt.step()  # update weights
            scheduler.step()
            # calculate loss to show the user
            avg_loss += loss / len(data_tr)
        toc = time()
        print('loss: %f' % avg_loss)
        losses['train'].append(avg_loss.item())

        # show intermediate results
        model.eval()  # testing mode
        avg_loss = 0
        for X_val, Y_val in data_val:
            with torch.no_grad():
              Y_hat = model(X_val).detach()
              loss =  loss_fn(Y_val, Y_hat)
              avg_loss += loss / len(data_tr)

        losses['valid'].append(avg_loss.item())
        scores.append(score_model(model, metric, data_val))
    return [losses, scores]
