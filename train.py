# imports
import os, argparse, pathlib
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


# import Augmentate
import data
import utils
import eval
from model_covid import CovidNet

def save_model(args_dict, state):
    """
    Saves model
    """
    directory = args_dict.dir_model
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'best_model.pth.tar'
    torch.save(state, filename)

def resume(args_dict, model, optimizer):
    """
    Continue training from a checkpoint
    :return: args_dict, model, optimizer from the checkppoint
    """
    best_sensit = -float('Inf')
    args_dict.start_epoch = 0
    if args_dict.resume:
        if os.path.isfile(args_dict.resume):
            print("=> loading checkpoint '{}'".format(args_dict.resume))
            checkpoint = torch.load(args_dict.resume)
            args_dict.start_epoch = checkpoint['epoch']
            best_sensit = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args_dict.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args_dict.resume))
            best_sensit = -float('Inf')

    return best_sensit, model, optimizer


def valEpoch(args_dict, dl_test, model):

    # switch to evaluation mode
    model.eval()
    for batch_idx, (x_batch, y_batch, _) in enumerate(dl_test):
        x_batch, y_batch = x_batch.to(args_dict.device), y_batch.to(args_dict.device)
        y_hat = np.argmax(model(x_batch).cpu().data.numpy(), axis=1)
        # Save embeddings to compute metrics
        if batch_idx == 0:
            pred = y_hat
            y_test = y_batch.cpu().data.numpy()
        else:
            pred = np.concatenate((pred, y_hat))
            y_test = np.concatenate((y_test, y_batch.cpu().data.numpy()))

    return eval.create_metrics(y_test, pred)

def trainEpoch(args_dict, dl_non_covid, dl_covid, model, criterion, optimizer, epoch):
    # object to store & plot the losses
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (x_batch_nc, y_batch_nc, weights_nc) in enumerate(dl_non_covid):
        x_batch_c, y_batch_c, weights_c = next(iter(dl_covid))

        x_batch = torch.cat((x_batch_nc, x_batch_c)).to(args_dict.device)
        y_batch = torch.cat((y_batch_nc, y_batch_c)).to(args_dict.device)
        weights = torch.cat((weights_nc, weights_c)).to(args_dict.device)  # What should we do with it?

        # Model output
        output = model(x_batch)

        # Loss
        train_loss = criterion(output, y_batch)
        losses.update(train_loss.data.cpu().numpy(), x_batch[0].size(0))

        # Accuracy
        max_indices = torch.max(output, axis=1)[1]
        train_acc = (max_indices == y_batch).sum().item() / max_indices.size()[0]
        accuracies.update(train_acc, x_batch[0].size(0))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})\t'.format(
               epoch, batch_idx, len(dl_non_covid), 100. * batch_idx / len(dl_non_covid),
               loss=losses, accuracy=accuracies))

        # Debug
        if batch_idx == 10:
            break

    # Plot loss
    # plotter.plot('loss', 'train', 'Cross Entropy Loss', epoch, losses.avg)
    # plotter.plot('Acc', 'train', 'Accuracy', epoch, accuracies.avg)

def train_model(args_dict):

    # Define model
    model = CovidNet(args_dict.n_classes)
    model.to(args_dict.device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args_dict.lr)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(args_dict.class_weights).to(args_dict.device))

    # Resume training if needed
    best_sensit, model, optimizer = resume(args_dict, model, optimizer)

    # Load data

    # Augmentation
    train_transformation = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.ColorJitter(brightness=(0.9, 1.1)),
        transforms.RandomRotation((-10,10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0,translate=(0.1,0.1), shear=10, scale=(0.85, 1.15), fillcolor=0),
        #TransformShow(), # visualize transformed pic
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.ToTensor()                               # to pytorch tensor
    ])

    # Dataloaders for training and validation
    # preprocess the given txt files: Train
    datasets_train, _, labels, labels_non, labels_cov = data.preprocessSplit(args_dict.train_txt)

    # create Datasets
    train_non_covid = data.Dataset(datasets_train[0], labels_non, args_dict.train_folder, transform=train_transformation)
    train_covid = data.Dataset(datasets_train[1], labels_cov, args_dict.train_folder, transform=train_transformation)

    covid_size = max(int(args_dict.batch * args_dict.covid_percent), 1)

    # create data loader
    dl_non_covid = DataLoader(train_non_covid, batch_size=(args_dict.batch - covid_size), shuffle=True)  # num_workers= 2
    dl_covid = DataLoader(train_covid, batch_size=covid_size, shuffle=True)  # num_workers= 2


    # Data loading for test
    # preprocess the given txt files: Test
    _, data_test, labels_test, _, _ = data.preprocessSplit(args_dict.test_txt)
    # create Datasets
    test_dataset = data.Dataset(data_test, labels_test, args_dict.test_folder, transform=val_transforms)
    # create data loader
    dl_test = DataLoader(test_dataset, batch_size=args_dict.batch, shuffle=False,  num_workers=1)


    # Now, let's start the training process!
    print('Start training...')
    pat_track = 0
    for epoch in range(args_dict.epochs):

        # Compute a training epoch
        trainEpoch(args_dict, dl_non_covid, dl_covid, model, criterion, optimizer, epoch)

        # Compute a validation epoch
        sensitivity_covid, accuracy = valEpoch(args_dict, dl_test, model)

        # TODO: implement the patience stop
        # check patience
        # if accval >= best_val:
        #     pat_track += 1
        # else:
        #     pat_track = 0
        # if pat_track >= args_dict.patience:
        #     args_dict.freeVision = args_dict.freeComment
        #     args_dict.freeComment = not args_dict.freeVision
        #     optimizer.param_groups[0]['lr'] = args_dict.lr * args_dict.freeComment
        #     optimizer.param_groups[1]['lr'] = args_dict.lr * args_dict.freeVision
        #     print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
        #     print('Initial vision lr: %f' % optimizer.param_groups[1]['lr'])
        #     print('Initial classifier lr: %f' % optimizer.param_groups[2]['lr'])
        #     args_dict.patience = 3
        #     pat_track = 0

        # TODO: save the model in case of a better sensitivity
        # save if it is the best model
        if accuracy >= 0.75:  # only compare sensitivity if we have a minimum accuracy of 0.8
            is_best = sensitivity_covid > best_sensit
            if is_best:
                best_sensit = max(sensitivity_covid, best_sensit)
                save_model(args_dict, {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_sensit': best_sensit,
                    'optimizer': optimizer.state_dict(),
                    'valtrack': pat_track,
                    # 'freeVision': args_dict.freeVision,
                    'curr_val': accuracy,
                })
        print('** Validation: %f (best_sensitivity) - %f (current acc) - %d (patience)' % (best_sensit, accuracy,
                                                                                           pat_track))

        # Plot
        # plotter.plot('Sensitivity', 'test', 'sensitivity covid', epoch, sensitivity_covid)
        # plotter.plot('Accuracy', 'test', 'Accuracy', epoch, accuracy)

def run_train(args_dict):
    # Set seed for reproducibility
    torch.manual_seed(args_dict.seed)

    # Set up device
    if torch.cuda.is_available():
        args_dict.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        args_dict.device = torch.device("cpu")
        print("Running on the CPU")

    # Plots
    # global plotter
    # plotter = utils.VisdomLinePlotter(env_name=args_dict.name)

    # Main process
    train_model(args_dict)
