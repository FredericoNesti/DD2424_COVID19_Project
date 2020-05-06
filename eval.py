import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from data import *
from torch.utils.data import DataLoader

def train(test_txt, test_folder, batch):
    # Data loading for test

    # preprocess the given txt files: Test
    _, data_test, labels_test, _, _ = preprocessSplit(test_txt)
    # create Datasets
    test_dataset = Dataset(data_test, labels_test, test_folder, transform=None)
    # create data loader
    dl_test = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=1)

    net = Net()
    net.load_state_dict(torch.load(PATH))

def create_metrics(matrix):
    '''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        matrix = confusion_matrix(pred.view(-1), target.view(-1)) # https://discuss.pytorch.org/t/confusion-matrix/21026/7
        # taken from COVIDNET
        matrix = matrix.astype('float')
        # cm_norm = matrix / matrix.sum(axis=train)[:, np.newaxis]
        print(matrix)
        # class_acc = np.array(cm_norm.diagonal())
        '''

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    # cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
    print(matrix)

    class_acc = [matrix[i, i] / np.sum(matrix[i, :]) if np.sum(matrix[i, :]) else 0 for i in range(len(matrix))]
    print('Sens Pneumonia: {0:.3f}, Normal: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i, i] / np.sum(matrix[:, i]) if np.sum(matrix[:, i]) else 0 for i in range(len(matrix))]
    print('PPV Pneumonia: {0:.3f}, Normal: {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))



