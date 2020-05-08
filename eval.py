import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from data import *
from torch.utils.data import DataLoader
from model_covid import CovidNet
from sklearn.metrics import accuracy_score


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