
from Dataloader import *
import argparse
import time

parser = argparse.ArgumentParser(description='COVID19')
parser.add_argument('--dataset', type=str, default='', metavar='N', help='')
parser.add_argument('--device', type=str, default='cuda', metavar='N', help='')
parser.add_argument('--batch', type=int, default='', metavar='N', help='')
parser.add_argument('--splitset', nargs='+',type=int, default=[0.6,0.1,0.3], metavar='N',help='')
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()

    COVID19_database = DATABASE(batch=args.batch,
                                split=args.splitset,
                                device=args.device)

    train_set, valid_set, test_set = COVID19_database.split_database()

    elapsed_time = time.time() - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


