from torch.utils import data
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib.pyplot import imshow

# taken from covid net
def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files


def preprocessSplit(csv_file):
    dataset = _process_csv_file(csv_file)

    datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    pictures = []
    labels = []
    for l in dataset:
        datasets[l.split()[2]].append(l.split()[1]) # just save the image
        pictures.append(l.split()[1])
        labels.append(l.split()[2])
    datasets = [
        datasets['normal'] + datasets['pneumonia'],
        datasets['COVID-19'],
    ]
    print(len(datasets[0]), len(datasets[1]))
    return datasets, pictures, labels



class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, datadir, transform=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.datadir = datadir
        self.input_shape = (224, 224)
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')

        # data loading taken from covid net
        X = cv2.imread(os.path.join(self.datadir, ID))
        h, w, c = X.shape
        X = X[int(h / 6):, :]
        X = cv2.resize(X, self.input_shape)

        if self.transform: # see https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format
            img = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
            print(img.shape)
            im_pil = Image.fromarray(img)

            imshow(np.asarray(im_pil))
            X = self.transform(im_pil)

        y = self.labels[index]

        return X, y