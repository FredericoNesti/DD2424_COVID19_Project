import time

from train import run_train
from eval import run_test
from calibration import run_calibration, plot_calibration
from params import get_parser
from pred import run_prediction

if __name__ == "__main__":
    start_time = time.time()

    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    assert args_dict.model in ['covidnet', 'resnet'], 'incorrect model selection! --model best be covidnet or resnet'

    if args_dict.mode == 'train':
        run_train(args_dict)

    elif args_dict.mode == 'test':
        run_test(args_dict)

    elif args_dict.mode == 'predict':
        run_prediction(args_dict)

    elif args_dict.mode == 'calibration':
        # run_calibration(args_dict)
        plot_calibration(args_dict)

    elapsed_time = time.time() - start_time
    time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
