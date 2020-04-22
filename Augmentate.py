

from torchvision import transforms as tf
from torchvision.datasets import ImageFolder

def Augmentation(t=(0,1),b=(0,1),d=0,c=(0,0),s=(0,0),h=(0,0)):
    transformation = tf.Compose([
        tf.RandomAffine(degrees=d, translate=t),
        tf.ColorJitter(brightness=b,contrast=c,saturation=s,hue=h),
        tf.RandomRotation(),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
    ])
    return transformation
