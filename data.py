from torch.utils import data
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms as tf

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

    labels_non = ['normal'] * len(datasets['normal']) + ['pneumonia'] * len(datasets['pneumonia'])
    labels_cov = ['COVID-19'] * len(datasets['COVID-19'])
    datasets = [
        datasets['normal'] + datasets['pneumonia'],
        datasets['COVID-19'],
    ]
    return datasets, pictures, labels, labels_non, labels_cov

# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
def make_weights_for_balanced_classes(labels,mapping, nclasses, defined_percentage):
    count = [0] * nclasses
    for item in labels:
        count[mapping[item]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[mapping[val]]
    return weight


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, datadir, transform=None, mapping={'normal':0, 'pneumonia':1, 'COVID-19':2 }, class_weights =[1., 1., 6.],  dimension = (224, 224) ):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.datadir = datadir
        self.input_shape = dimension
        self.transform = transform
        self.class_weights = class_weights
        self.mapping = mapping

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        #PIL loading
        image = Image.open(os.path.join(self.datadir, ID)).convert('RGB')
        im_pil = image.resize(self.input_shape)

        if self.transform: # see https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format
            X = self.transform(im_pil)
        else:
            transformer_tf = tf.ToTensor()
            X = transformer_tf(im_pil)

        y = self.labels[index]

        weights = np.take(self.class_weights, self.mapping[y])
        y = torch.tensor(self.mapping[y], dtype=torch.long)

        return X, y, weights