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

def valEpoch(args_dict, dl_test, model):

    # switch to evaluation mode
    model.eval()
    for batch_idx, (x_batch, y_batch, _) in enumerate(dl_test):
        x_batch, y_batch = x_batch.to(args_dict.device), y_batch.to(args_dict.device)
        output = model(x_batch)
        y_hat = np.argmax(output.cpu().data.numpy(), axis=1)
        # Save embeddings to compute metrics
        if batch_idx == 0:
            pred = y_hat
            y_test = y_batch.cpu().data.numpy()
        else:
            pred = np.concatenate((pred, y_hat))
            y_test = np.concatenate((y_test, y_batch.cpu().data.numpy()))

    if args_dict.model == 'resnet':
        grid_image, images = grad_cam(model, x_batch)
    elif args_dict.model == 'covidnet':
        output[:, pred].backward()
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(x_batch).detach()

        for i in range(2048):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        #heatmap = np.minimum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.min(heatmap)

        # draw the heatmap
        plt.matshow(heatmap.squeeze())

        apply_heatmap(heatmap, x_batch)





    #return create_metrics(y_test, pred)

def apply_heatmap(heatmap, x_batch):
    img = x_batch[0].numpy().transpose((1,2,0))
    #img = cv2.imread('./data/test/1/covid-19-rapidly-progressive-acute-respiratory-distress-syndrome-ards-day-2.jpg')
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_AREA)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imshow('map',superimposed_img)
    #cv2.imwrite('./map.jpg', superimposed_img)
    print('end')


# Grad-Cam implementation from:  https://github.com/vickyliin/gradcam_plus_plus-pytorch
def grad_cam(model, x_batch):
    images = []
    gradcam = GradCAM.from_config(arch=model._modules['resnet'], model_type='resnet', layer_name='7')
    mask, _ = gradcam(x_batch)
    heatmap, result = visualize_cam(mask, x_batch)
    images.extend([torch.reshape(x_batch, [3, 224, 224]).cpu(), heatmap, result])
    grid_image = make_grid(images, nrow=3)
    #plt.imshow(np.asarray(transforms.ToPILImage()(grid_image)))
    return grid_image, images

def run_test(args_dict):

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
    _, data_test, labels_test, _, _ = data.preprocessSplit(args_dict.test_txt)
    # create Datasets
    test_dataset = data.Dataset(data_test, labels_test, args_dict.test_folder, transform=val_transforms)
    # create data loader
    dl_test = DataLoader(test_dataset, batch_size=args_dict.batch, shuffle=False,  num_workers=1)

    valEpoch(args_dict, dl_test, model)


def create_metrics(y_test, pred):

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')

    print(matrix)

    sensitivity_covid = matrix[2,2] / (matrix[2,0]+matrix[2,1]+matrix[2,2])
    print("Sensitivity Covid19: ", sensitivity_covid)

    acc = accuracy_score(y_test, pred)
    print("Accuracy", acc)


    # From COVIDNET
    class_acc = [matrix[i, i] / np.sum(matrix[i, :]) if np.sum(matrix[i, :]) else 0 for i in range(len(matrix))]
    print('Sens Pneumonia: {0:.3f}, Normal: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i, i] / np.sum(matrix[:, i]) if np.sum(matrix[:, i]) else 0 for i in range(len(matrix))]
    print('PPV Pneumonia: {0:.3f}, Normal: {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                              ppvs[1],
                                                                              ppvs[2]))

    return sensitivity_covid, acc

