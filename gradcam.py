import torch
import numpy as np
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam

from torch.utils.data import DataLoader
from torchvision import transforms

import data
import utils
import eval
from model_covid import CovidNet, ResNet
import cv2
from matplotlib import pyplot as plt


# Grad-Cam implementation from:  https://github.com/vickyliin/gradcam_plus_plus-pytorch
def grad_cam(model, x_batch):
    gradcam = GradCAM.from_config(arch=model._modules['resnet'], model_type='resnet', layer_name='7')
    mask, _ = gradcam(x_batch)
    heatmap, result = visualize_cam(mask, x_batch)
    result =  result.numpy().transpose(1, 2, 0)
    return heatmap,result


def apply_heatmap(heatmap, x_batch, intesity=0.001):
    img = x_batch[0].numpy().transpose((1,2,0))
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * intesity + img
    return heatmap, superimposed_img


# Grad-Cam implementation from:  https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
def grad_cam_covid(model, x_batch, output, pred):
    output[:, pred].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(x_batch).detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    return apply_heatmap(heatmap, x_batch)


def run_gradcam(args_dict): # call this function to get the gradcam pictures and output - only for one picture
    print("Start test of model...")
    args_dict.batch = 1

    # Set up device
    if torch.cuda.is_available():
        args_dict.device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        args_dict.device = torch.device("cpu")
        print("Running on the CPU")

    args_dict.resume = True
    # Define model
    if args_dict.model == "covidnet":
        model = CovidNet(args_dict.n_classes)
    else:
        model = ResNet(args_dict.n_classes)

    model.to(args_dict.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_dict.lr)

    best_sensit, model, optimizer = utils.resume(args_dict, model, optimizer)

    dl_test = eval.calculateDataLoaderTest(args_dict)
    for batch_idx, (x_batch, y_batch, _) in enumerate(dl_test):
        x_batch, y_batch = x_batch.to(args_dict.device), y_batch.to(args_dict.device)
        if batch_idx == 2:
            break

    output = model(x_batch)
    pred = np.argmax(output.cpu().data.numpy(), axis=1)

    if args_dict.model == 'resnet':
        heatmap, image = grad_cam(model, x_batch)
    elif args_dict.model == 'covidnet':
        heatmap, image = grad_cam_covid(model, x_batch, output, pred)


    # plt.imshow(image, interpolation='nearest')
    print(heatmap.shape)
    print(image.shape)
    plt.imshow(heatmap)
    plt.show()

    return heatmap, image, output





