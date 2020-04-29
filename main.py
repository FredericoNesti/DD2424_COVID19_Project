import time
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from Augmentate import *
from data import *

parser = argparse.ArgumentParser(description='COVID19')

# training params
parser.add_argument('--device', type=str, default='cuda', metavar='N', help='')
parser.add_argument('--batch', type=int, default='', metavar='N', help='')
parser.add_argument('--epochs', type=int, default=1, metavar='N',help='')

# data sources
parser.add_argument('--train_folder', type=str, default='', metavar='N',help='')
parser.add_argument('--validation_folder', type=str, default='', metavar='N',help='')
parser.add_argument('--test_folder', type=str, default='', metavar='N',help='')
parser.add_argument('--test_txt', type=str, default='train_split_v3.txt', metavar='N',help='')


# results params
parser.add_argument('--results_directory', type=str, default='', metavar='N',help='')

args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    class_weights =[1., 1., 6.],# move to args
    mapping = {'normal':0, 'pneumonia':1, 'COVID-19':2 }


    datasets, pictures, labels = preprocessSplit(args.test_txt)
    weight_list = make_weights_for_balanced_classes(labels,mapping, 3, [0.375, 0.375, 0.25])
    training_set = Dataset(pictures, labels, args.train_folder, class_weights, mapping,  transform=Augmentation())
    train_loaded = DataLoader(training_set, batch_size=args.batch,  sampler=WeightedRandomSampler(weight_list, len(weight_list)))


    for e in range(1, args.epochs+1):
        print('Epoch: ', e)
        # training comes here
        for batch_idx, (inputs, y_batch, weights) in enumerate(train_loaded):
            print(y_batch)
            print(weights)

            if batch_idx == 99: # stop condition
                break



    elapsed_time = time.time() - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))




