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


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def train_my_model(model, criterion, optimizer, scheduler, verbose=False, epochs=10):
    acc_tr, acc_test, loss_tr = [], [], []
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        running_loss_train = 0.0
        train_acc_sum = 0.0
        test_acc_sum = 0.0

        if epoch == 0:
            for i, (x_test, y_test) in enumerate(testloader):
                print(i)
                x_test, y_test = x_test.to(device), y_test.to(device)
                y_hat_test = model(x_test)
                test_acc_sum += (y_hat_test.argmax(dim=1) == y_test).sum().item()
                torch.cuda.empty_cache()

            acc_tr.append(train_acc_sum / n_images_train)
            acc_test.append(test_acc_sum / n_images_test)

        for i, (x_batch, y_batch) in tqdm(enumerate(trainloader)):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss_train += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y_batch).sum().item()

        test_acc_sum = 0.0
        # print statistics
        for x_test, y_test in testloader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_hat_test = model(x_test)
            test_acc_sum += (y_hat_test.argmax(dim=1) == y_test).sum().item()

        acc_tr.append(train_acc_sum / n_images_train)
        acc_test.append(test_acc_sum / n_images_test)
        loss_tr.append(running_loss_train / n_images_train)

        if verbose:
            print('epoch: %d loss: %.4f  acc train: %.3f  acc test: %.3f' %
                  (epoch, loss_tr[-1], acc_tr[-1], acc_test[-1]))

    print('Finished Training')
    return acc_tr, acc_test, loss_tr

# normalization
trainloader, testloader, validloader, train_data = load_data('flowers')

n_images_train = 102
n_images_test = 50

net = CovidNet(10).to(device)
verbose = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), momentum=0.1, lr=0.05, weight_decay=0.002)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=10e-5, max_lr=0.1)

print(torch.cuda.get_device_properties(device).total_memory)

acc_tr_bn, acc_test_bn, loss_tr_bn = train_my_model(net, criterion, optimizer, scheduler, verbose)