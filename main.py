
from Dataloader import *
import argparse
import time

parser = argparse.ArgumentParser(description='COVID19')
parser.add_argument('--dataset', type=str, default='', metavar='N', help='')
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    train_set, valid_set, test_set =

    elapsed_time = time.time() - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

