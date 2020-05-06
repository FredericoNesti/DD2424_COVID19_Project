import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from model_covid import CovidNet
from utils_ic import load_data
from tqdm import tqdm

def train(model, criterion, optimizer, trainloader, testloader, n_images_train, n_images_test,
          device, verbose=False, epochs=10):
    """
        train the model for one epoch
    :param model:
    :param criterion:
    :param optimizer:
    :param trainloader:
    :param testloader:
    :param n_images_train:
    :param n_images_test:
    :param device:
    :param verbose:
    :param epochs:
    :return:
    """
    model.train()  # set model to training mode
    acc_tr, acc_test, loss_tr = [], [], []

    running_loss_train = 0.0
    train_acc_sum = 0.0

    for i, (x_batch, y_batch) in tqdm(enumerate(trainloader)):
        batch_size = x_batch.shape[0]
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        loss.backward()
        optimizer.step()

        running_loss_train += loss.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y_batch).sum().item()

        acc_tr.append(train_acc_sum / batch_size)
        loss_tr.append(running_loss_train / batch_size)

        if verbose:
            print("loss: {:.4}  acc train: {:.4}".format(loss_tr[-1], acc_tr[-1]))

    print('Finished Training')
    return acc_tr, loss_tr


def main():
    learning_rate = 2e-5
    n_epochs = 22
    batch_size = 16
    factor = 0.7
    patience = 5

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # generate dataset
    trainloader, testloader, validloader, train_data = load_data('flowers', batch_size=batch_size)

    n_images_train = 6755
    n_images_test = 1022

    model = CovidNet(102).to(device)
    verbose = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=10e-5, max_lr=0.1)
    # Adam does not support scheduler

    print("Number of parameters of the model: {:.4} M".format(model.get_n_params() / 1e6))

    acc_tr_bn, loss_tr_bn = train(model, criterion, optimizer, trainloader,testloader,
                                  n_images_train, n_images_test, device, verbose=verbose, epochs=n_epochs)

    print("finished")


if __name__ == '__main__':
    main()