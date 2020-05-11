import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='COVID19')

    parser.add_argument('--name', type=str, default='covid', metavar='N', help='name of the environment')
    # training params
    parser.add_argument('--device', type=str, default='cuda', metavar='N', help='')
    parser.add_argument('--mode', type=str, default='train', metavar='N', help='[train, test]')
    parser.add_argument('--seed', type=int, default=1234, metavar='N', help='seed for reproducibility')

    # Model params
    parser.add_argument('--dir_model', type=str, default='models/', metavar='N')
    parser.add_argument('--model', type=str, default="covidnet", metavar='N', help='Model [covidnet , resnet]')
    parser.add_argument('--n_classes', type=str, default=3, metavar='N', help='Number of classes of the output')
    parser.add_argument('--batch', type=int, default=8, metavar='N', help='')
    parser.add_argument('--epochs', type=int, default=4, metavar='N', help='')
    parser.add_argument('--lr', type=str, default=2e-5, metavar='N', help='Learning rate')
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--class_weights', default=[1., 1., 6.], type=list, help='weights for loss for each class')
    parser.add_argument('--resume', default=False, type=bool)


    # data sources
    parser.add_argument('--train_folder', type=str, default='data/train', metavar='N', help='')
    parser.add_argument('--test_folder', type=str, default='data/test', metavar='N', help='')
    parser.add_argument('--train_txt', type=str, default='data/train_split_v3.txt', metavar='N', help='')
    parser.add_argument('--test_txt', type=str, default='data/test_split_v3.txt', metavar='N', help='')
    parser.add_argument('--covid_percent', type=float, default=0.3, metavar='N', help='')

    # results params
    parser.add_argument('--results_directory', type=str, default='', metavar='N', help='')


    return parser