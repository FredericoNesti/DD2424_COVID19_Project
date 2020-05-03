from torchvision import transforms as tf
import cv2
import numpy as np

def Augmentation(r=(-10,10), t=(0.1,0.1),shear=10, scale=(0.85, 1.15) ,b=(0.9, 1.1),cval=0):
    transformation = tf.Compose([
        #tf.Resize((224, 224)),
        tf.ColorJitter(brightness=b),
        tf.RandomRotation(r, fill=cval),
        tf.RandomHorizontalFlip(),
        tf.RandomAffine(0,translate=t, shear=shear, scale=scale, fillcolor=cval),
        #TransformShow(), # visualize transformed pic
        tf.ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] i
    ])
    return transformation

# for visualizing the transformations: taken from https://stackoverflow.com/questions/55179282/display-examples-of-augmented-images-in-pytorch
def TransformShow(name="transformed", wait=100):
    def transform_show(img):
        cv2.imshow(name, np.array(img))
        cv2.waitKey(wait)
        return img
    return transform_show