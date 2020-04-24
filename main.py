import time
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from Augmentate import *

parser = argparse.ArgumentParser(description='COVID19')

# training params
parser.add_argument('--device', type=str, default='cuda', metavar='N', help='')
parser.add_argument('--batch', type=int, default='', metavar='N', help='')
parser.add_argument('--epochs', type=int, default=1, metavar='N',help='')

# data sources
parser.add_argument('--train_folder', type=str, default='', metavar='N',help='')
parser.add_argument('--validation_folder', type=str, default='', metavar='N',help='')
parser.add_argument('--test_folder', type=str, default='', metavar='N',help='')

# results params
parser.add_argument('--results_directory', type=str, default='', metavar='N',help='')

args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    datasets, pictures, labels = preprocessSplit('train_split_v3.txt')
    #print(labels)
    training_set = Dataset(pictures, labels, 'data/train/1/', transform=Augmentation())
    train_loaded = DataLoader(training_set, batch_size=args.batch, shuffle=True)

    #X = cv2.imread(os.path.join('data/train/1/', 'pneumococcal-pneumonia-day0.jpg'))
    #print(X)

    '''

    train_dataset = ImageFolder(root=args.train_folder, transform=Augmentation())
    train_loaded = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    val_dataset = ImageFolder(root=args.validation_folder, transform=Augmentation())
    val_loaded = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)

    # test dataset with augmentation
    test_dataset_A = ImageFolder(root=args.test_folder, transform=Augmentation())
    test_loaded_A = DataLoader(test_dataset_A, batch_size=args.batch, shuffle=True)

    # test dataset without augmentation
    test_dataset_N = ImageFolder(root=args.test_folder, transform=ToTensor())
    test_loaded_N = DataLoader(test_dataset_N, batch_size=args.batch, shuffle=True)
'''
    for e in range(1, args.epochs+1):
        print('Epoch: ', e)
        # training comes here

        for batch_idx, (inputs, y_batch) in enumerate(train_loaded):
            print(inputs)
            print(y_batch)



    elapsed_time = time.time() - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))




