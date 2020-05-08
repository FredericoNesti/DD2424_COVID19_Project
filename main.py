import time
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler

from train import run_train
# from test import run_test
from params import get_parser


if __name__ == "__main__":
    start_time = time.time()

    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()



    if args_dict.mode == 'train':
        run_train(args_dict)

    elif args_dict.mode == 'test':
        # run_test(args_dict)
        print("todo")

    elapsed_time = time.time() - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))




# if args.mode == 'train':
#     # preprocess the given txt files: Train
#     datasets, _, _, labels_non, labels_cov = preprocessSplit(args.test_txt)
#     # create Datasets
#     train_non_covid = Dataset(datasets[0], labels_non, args.train_folder,  transform=Augmentation())
#     train_covid = Dataset(datasets[1], labels_cov , args.train_folder, transform=Augmentation())
#     covid_size = max(int(args.batch * args.covid_percent), 1)
#     # create data loader
#     dl_non_covid = DataLoader(train_non_covid, batch_size=(args.batch-covid_size), shuffle=True) # num_workers= 2
#     dl_covid = DataLoader(train_covid, batch_size=covid_size, shuffle=True) # num_workers= 2
# else:
#     # preprocess the given txt files: Test
#     ''' Should we also balance the testing?'''
#     _, data_test, labels_test, _, _ = preprocessSplit(args.test_txt)
#     # create Datasets
#     test_dataset = Dataset(data_test, labels_test, args.test_folder, transform=None)
#     # create data loader
#     dl_test = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, num_workers=1)

    # else:
    #     for batch_idx, (x_batch, y_batch, weights) in enumerate(dl_test):
    #
    #         print(x_batch.shape)
    #         print(y_batch)
    #         print(weights)
    #
    #         if batch_idx == 99: # stop condition
    #             break