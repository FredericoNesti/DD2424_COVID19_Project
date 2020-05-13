import torch
import numpy as np
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam

from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

import data
import utils
from model_covid import CovidNet, ResNet
import cv2
from PIL import Image



# Grad-Cam implementation from:  https://github.com/vickyliin/gradcam_plus_plus-pytorch
def grad_cam(model, x_batch):
    gradcam = GradCAM.from_config(arch=model._modules['resnet'], model_type='resnet', layer_name='7')
    mask, _ = gradcam(x_batch)
    heatmap, result = visualize_cam(mask, x_batch)
    result = transforms.ToPILImage()(result)
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

def run_prediction(args_dict): # for testing the graca -- dataloader with only 1 image

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
    else:
        model = ResNet(args_dict.n_classes)

    model.to(args_dict.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_dict.lr)

    best_sensit, model, optimizer = utils.resume(args_dict, model, optimizer)

    val_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.ToTensor()                               # to pytorch tensor
    ])

    # Data loading for test
    # preprocess the given txt files: Test
    _, data_pred, labels_pred, _, _ = data.preprocessSplit(args_dict.predict_txt)


    # create Datasets
    pred_dataset = data.Dataset(data_pred, labels_pred, args_dict.test_folder, transform=val_transforms)
    # create data loader
    dl_pred = DataLoader(pred_dataset, batch_size=args_dict.batch, shuffle=False,  num_workers=1)

    # switch to evaluation mode
    model.eval()

    for batch_idx, (x_batch, y_batch, _) in enumerate(dl_pred):
        x_batch, y_batch = x_batch.to(args_dict.device), y_batch.to(args_dict.device)

    predict(x_batch, model, args_dict)

def predict(x_batch, model, args_dict): # call this function to get the gradcam pictures and output - only for one picture
    output = model(x_batch)
    pred = np.argmax(output.cpu().data.numpy(), axis=1)

    if args_dict.model == 'resnet':
        heatmap, image = grad_cam(model, x_batch)
        image.show()
        image.save('./resnetpic.jpg')
    elif args_dict.model == 'covidnet':
        heatmap, image = grad_cam_covid(model, x_batch, output, pred)
        cv2.imshow('map', image)
        image = cv2.convertScaleAbs(image, alpha=(255.0))
        cv2.imwrite('./covidpic.jpg', image)

    return heatmap, image, output





