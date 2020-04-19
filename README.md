# Todo

## Setup infrastructure
- sign up and use the credits (svenja sign up, Matheus setup project)

## Data part --> fred implement and see if it can be split up and tell us
- preprocessing the data (take from covnet) - done in create_COVIDx_v3.ipynb (Covid Net)

- data loader + augmentation (PyTorch)
- balancing data - done in data.py (Covid Net)
- augmentation: translation, rotation, horizontal flip, intensity shift

## Network architecture (Matheus)
- 

## Training (Svenja)
- pretrained on ImageNet dataset
- trained on COVIDx dataset using g the Adam optimizer using a learning rate policy where the learning rate decreases when learning  stagnates for a period of time (i.e., ’patience’)
-  The following hyperparameters were used for training: learning rate=2e-5, number
of epochs=22, batch size=8, factor=0.7, patience=5
- data augmentation was leveraged with the following augmentation types: translation, rotation, horizontal flip, and intensity shift
- batch re-balancing strategy to promote better distribution of each infection type at a batch level

## Evaluating results (Svenja)
### Quantitative: 
-  computed the test accuracy, as well as sensitivity and positive predictive value (PPV) for each infection type, on the aforementioned COVIDx dataset
-  sensitivity and PPV for each infection type
- confusion matrix 


## Improvenments / Changes to try out
- loss function 
- augmentation techniques
