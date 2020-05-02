import time
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler

from Augmentate import *
from data import *

parser = argparse.ArgumentParser(description='COVID19')

# training params
parser.add_argument('--device', type=str, default='cuda', metavar='N', help='')
parser.add_argument('--batch', type=int, default='', metavar='N', help='')
parser.add_argument('--epochs', type=int, default=1, metavar='N',help='')
parser.add_argument('--mode', type=str, default='train', metavar='N',help='')
parser.add_argument('--covid_percent', type=float, default=0.3, metavar='N',help='')

# data sources
parser.add_argument('--train_folder', type=str, default='', metavar='N',help='')
parser.add_argument('--test_folder', type=str, default='', metavar='N',help='')
parser.add_argument('--test_txt', type=str, default='train_split_v3.txt', metavar='N',help='')
parser.add_argument('--text_txt', type=str, default='test_split_v3.txt', metavar='N',help='')

# results params
parser.add_argument('--results_directory', type=str, default='', metavar='N',help='')

args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    ''' removed for first approach
    weight_list = make_weights_for_balanced_classes(labels,mapping, 3, [0.375, 0.375, 0.25])
    train_loaded = DataLoader(training_set, batch_size=args.batch,  sampler=WeightedRandomSampler(weight_list, len(weight_list)))
    training_set = Dataset(pictures, labels, args.train_folder, class_weights,  transform=Augmentation())
    '''

    if args.mode == 'train':
        # preprocess the given txt files: Train
        datasets, _, _, labels_non, labels_cov = preprocessSplit(args.test_txt)

        # create Datasets
        train_non_covid = Dataset(datasets[0], labels_non, args.train_folder,  transform=Augmentation())
        train_covid = Dataset(datasets[1], labels_cov , args.train_folder, transform=Augmentation())
        covid_size = max(int(args.batch * args.covid_percent), 1)

        dl_non_covid = DataLoader(train_non_covid, batch_size=(args.batch-covid_size), shuffle=True, num_workers= 2)
        dl_covid = DataLoader(train_covid, batch_size=covid_size, shuffle=True, num_workers= 2)
    else:
        # preprocess the given txt files: Test
        ''' Should we also balance the testing?'''
        _, data_test, labels_test, _, _ = preprocessSplit(args.test_txt)
        # create Datasets
        test_dataset = Dataset(data_test, labels_test, args.test_folder)
        dl_test = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=1)



    if args.mode == 'train':
        for batch_idx, (x_batch_nc, y_batch_nc, weights_nc) in enumerate(dl_non_covid):
            x_batch_c, y_batch_c, weights_c = next(iter(dl_covid))

            x_batch = torch.cat((x_batch_nc, x_batch_c))
            y_batch = torch.cat((y_batch_nc , y_batch_c))
            weights = torch.cat((weights_nc, weights_c))

            print(x_batch.shape)
            print(y_batch)
            print(weights)
            if batch_idx == 99: # stop condition
                break

    else:
        for batch_idx, (x_batch, y_batch, weights) in enumerate(dl_test):

            print(x_batch.shape)
            print(y_batch)
            print(weights)

            if batch_idx == 99: # stop condition
                break




    elapsed_time = time.time() - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))




