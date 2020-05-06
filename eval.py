import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from data import *
from torch.utils.data import DataLoader
from model_covid import CovidNet
from sklearn.metrics import accuracy_score


############### added for debugging
import torch.nn as nn
import torch.nn.functional as F
class OwnNet(nn.Module):
    def __init__(self, n_classes=3):
        super(OwnNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6 , 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc1 = nn.Linear(179776, 120)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 179776)
        x = x.view(x.shape[0], 44944)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test(test_txt, test_folder, batch, device, path):
    y_test = []
    pred = []

    # Data loading for test
    # preprocess the given txt files: Test
    _, data_test, labels_test, _, _ = preprocessSplit(test_txt)
    # create Datasets
    test_dataset = Dataset(data_test, labels_test, test_folder, transform=None)
    # create data loader
    dl_test = DataLoader(test_dataset, batch_size=batch, shuffle=False,  num_workers=1)

    #model = CovidNet(102).to(device)
    #model.load_state_dict(torch.load(path))
    model = OwnNet().to(device)

    for batch_idx, (x_batch, y_batch, _) in enumerate(dl_test):
        y_test.append(y_batch)
        pred.append(np.argmax(model(x_batch).detach().numpy(), axis=1))
        if batch_idx == 1000:
            break

    y_test = np.array(y_test)
    pred = np.array(pred)

    return y_test, pred

def create_metrics(y_test, pred):

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    # cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)

    print("Accuracy",accuracy_score(y_test, pred))

    # From COVIDNET
    class_acc = [matrix[i, i] / np.sum(matrix[i, :]) if np.sum(matrix[i, :]) else 0 for i in range(len(matrix))]
    print('Sens Pneumonia: {0:.3f}, Normal: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i, i] / np.sum(matrix[:, i]) if np.sum(matrix[:, i]) else 0 for i in range(len(matrix))]
    print('PPV Pneumonia: {0:.3f}, Normal: {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))

