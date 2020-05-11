import torch
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

import data
import utils
from model_covid import CovidNet, ResNet

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

    return create_metrics(y_test, pred)

def run_test(args_dict):

    # Set up device
    if torch.cuda.is_available():
        args_dict.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        args_dict.device = torch.device("cpu")
        print("Running on the CPU")

    # Define model
    if args_dict.model == "covidnet":
        model = CovidNet(args_dict.n_classes)
    else:
        model = ResNet(args_dict.n_classes)

    model.to(args_dict.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_dict.lr)

    best_sensit, model, optimizer = utils.resume(args_dict, model, optimizer)

    val_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.ToTensor()                               # to pytorch tensor
    ])

    # Data loading for test
    # preprocess the given txt files: Test
    _, data_test, labels_test, _, _ = data.preprocessSplit(args_dict.test_txt)
    # create Datasets
    test_dataset = data.Dataset(data_test, labels_test, args_dict.test_folder, transform=val_transforms)
    # create data loader
    dl_test = DataLoader(test_dataset, batch_size=args_dict.batch, shuffle=False,  num_workers=1)

    valEpoch(args_dict, dl_test, model)


def create_metrics(y_test, pred):

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')

    print(matrix)

    sensitivity_covid = matrix[2,2] / (matrix[2,0]+matrix[2,1]+matrix[2,2])
    print("Sensitivity Covid19: ", sensitivity_covid)

    acc = accuracy_score(y_test, pred)
    print("Accuracy", acc)


    # From COVIDNET
    class_acc = [matrix[i, i] / np.sum(matrix[i, :]) if np.sum(matrix[i, :]) else 0 for i in range(len(matrix))]
    print('Sens Pneumonia: {0:.3f}, Normal: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i, i] / np.sum(matrix[:, i]) if np.sum(matrix[:, i]) else 0 for i in range(len(matrix))]
    print('PPV Pneumonia: {0:.3f}, Normal: {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                              ppvs[1],
                                                                              ppvs[2]))

    return sensitivity_covid, acc

