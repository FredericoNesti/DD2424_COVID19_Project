import torch

import utils
import eval
from sklearn.calibration import calibration_curve

from model_covid import CovidNet, ResNet
from temperature_scaling import ModelWithTemperature


def plot_calibration(args_dict):

    # Set up device
    if torch.cuda.is_available():
        args_dict.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    else:
        args_dict.device = torch.device("cpu")

    dl_test = eval.calculateDataLoaderTest(args_dict)

    # Define model
    if args_dict.model == "covidnet":
        model_normal = CovidNet(args_dict.n_classes)

    elif args_dict.model == "resnet":
        model_normal = ResNet(args_dict.n_classes)

    model_calib = ModelWithTemperature(model_normal)

    normal_model_path = args_dict.dir_model + args_dict.model + '_best_model.pth.tar'
    checkpoint_normal = torch.load(normal_model_path)
    model_normal.load_state_dict(checkpoint_normal['state_dict'])

    calib_model_path = args_dict.dir_model + "calibrated_" + args_dict.model + '_best_model.pth.tar'
    checkpoint_calib = torch.load(calib_model_path)
    model_calib.load_state_dict(checkpoint_calib['state_dict'])

    print("Calculating probabilities for test set...")
    probs, y_true = eval.valEpoch(args_dict, dl_test, model_normal, calibration=True)

    print("calibration graph...")
    
    calibration_curve(y_true, probs)

def run_calibration(args_dict):
    """
    Apply Temperature  scaling for callibration and saves the new model
    """

    print("Start calibration of model...")

    # Set up device
    if torch.cuda.is_available():
        args_dict.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        args_dict.device = torch.device("cpu")
        print("Running on the CPU")

    # Define model
    if args_dict.model == "covidnet":
        model = CovidNet(args_dict.n_classes)
    elif args_dict.model == "resnet":
        model = ResNet(args_dict.n_classes)

    model.to(args_dict.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_dict.lr)

    best_sensit, model, optimizer = utils.resume(args_dict, model, optimizer)

    dl_test = eval.calculateDataLoaderTest(args_dict)


    best_sensit, model, optimizer = utils.resume(args_dict, model, optimizer)

    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(dl_test, args_dict.device)


    utils.save_model(args_dict, {
        'epoch': args_dict.start_epoch,
        'state_dict': scaled_model.state_dict(),
        'optimizer': optimizer.state_dict()
    })

    # TODO: plot for calibration




