import torch

import utils
import eval
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
from scipy.special import softmax

from model_covid import CovidNet, ResNet
from temperature_scaling import ModelWithTemperature


def plot_calibration(args_dict):
    args_dict.resume = True

    dl_test = eval.calculateDataLoaderTest(args_dict)

    # Define model
    if args_dict.model == "covidnet":
        model_normal = CovidNet(args_dict.n_classes)

    elif args_dict.model == "resnet":
        model_normal = ResNet(args_dict.n_classes)

    # Set up device
    if torch.cuda.is_available():
        args_dict.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    else:
        args_dict.device = torch.device("cpu")

    # load normal model
    model_normal.to(args_dict.device)
    optimizer = torch.optim.Adam(model_normal.parameters(), lr=args_dict.lr)
    _, model_normal, _ = utils.resume(args_dict, model_normal, optimizer)

    # load calibrated model
    model_calib = ModelWithTemperature(model_normal)
    calib_model_path = args_dict.dir_model + "calibrated_" + args_dict.model + '_best_model.pth.tar'
    checkpoint_calib = torch.load(calib_model_path, map_location=torch.device(args_dict.device))
    model_calib.load_state_dict(checkpoint_calib['state_dict'])

    print("Calculating probabilities for test set...")
    probs_normal, y_true = eval.valEpoch(args_dict, dl_test, model_normal, calibration=True)
    probs_normal = softmax(probs_normal, axis=1)

    probs_calib, y_true = eval.valEpoch(args_dict, dl_test, model_calib, calibration=True)
    probs_calib = softmax(probs_calib, axis=1)

    print("calibration graph...")
    idx2class = {
        0: 'normal',
        1: 'pneumonia',
        2: 'COVID19'
    }

    fig, axs = plt.subplots(1, args_dict.n_classes, figsize=(15, 5))
    for idx_class in range(args_dict.n_classes):
        y_class = y_true == idx_class

        # reliability diagram
        fop_normal, mpv_normal = calibration_curve(y_class, probs_normal[:,idx_class])
        fop_calib, mpv_calib = calibration_curve(y_class, probs_calib[:, idx_class])
        # plot perfectly calibrated
        axs[idx_class].plot([0, 1], [0, 1], linestyle='--')
        # plot calibrated reliability
        axs[idx_class].plot(mpv_calib, fop_calib, marker='.', label='calibrated')
        axs[idx_class].plot(mpv_normal, fop_normal, marker='*', label='normal')
        axs[idx_class].set(xlabel='confidence')
        axs[idx_class].set(ylabel='accuracy')
        # title
        axs[idx_class].set_title(idx2class[idx_class])

    for ax in axs.flat:
        ax.label_outer()

    fig.autofmt_xdate()
    plt.subplots_adjust(wspace=0.1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',  ncol=args_dict.n_classes)
    fig.savefig('calibration_' + args_dict.model + '.png')


def run_calibration(args_dict):
    """
    Apply Temperature  scaling for callibration and saves the new model
    """

    print("Start calibration of model...")

    args_dict.resume = True

    # Define model
    if args_dict.model == "covidnet":
        model = CovidNet(args_dict.n_classes)
    elif args_dict.model == "resnet":
        model = ResNet(args_dict.n_classes)

    # Set up device
    if torch.cuda.is_available():
        args_dict.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        args_dict.device = torch.device("cpu")
        print("Running on the CPU")

    model.to(args_dict.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_dict.lr)

    dl_test = eval.calculateDataLoaderTest(args_dict)

    _, model, _ = utils.resume(args_dict, model, optimizer)
    # model.eval()

    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(dl_test, args_dict.device)

    print("saving calibrated model")
    utils.save_model(args_dict, {
        'epoch': args_dict.start_epoch,
        'state_dict': scaled_model.state_dict(),
        'optimizer': optimizer.state_dict()
    })

    plot_calibration(args_dict)
