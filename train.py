# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision.datasets import ImageFolder
from barbar import Bar
from torch.utils.data import DataLoader
import numpy as np
import os, argparse, pathlib
from Augmentate import *
# from eval import eval
# from data import BalanceCovidDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################
# model definition - example - todo matheus
# placeholder model#

class Net(nn.Module):
    def __init__(self, D_in=3, H=1285, D_out=1992): #fix this
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        D_in: input dimension
        H: dimension of hidden layer
        D_out: output dimension
        """
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
            """
            In the forward function we accept a Variable of input data and we must
            return a Variable of output data. We can use Modules defined in the
            constructor as well as arbitrary operators on Variables.
            """
            h_relu = F.relu(self.linear1(x))
            y_pred = self.linear2(h_relu)
            return y_pred

def loadData(train_folder, validation_folder, test_folder, batch=8  ):
    # load data
    parser = argparse.ArgumentParser(description='COVID19')

    train_dataset = ImageFolder(root=train_folder, transform=Augmentation())
    train_loaded = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    val_dataset = ImageFolder(root=validation_folder, transform=Augmentation())
    val_loaded = DataLoader(val_dataset, batch_size=batch, shuffle=True)

    # test dataset with augmentation
    test_dataset_A = ImageFolder(root=test_folder, transform=Augmentation())
    test_loaded_A = DataLoader(test_dataset_A, batch_size=batch, shuffle=True)

    # test dataset without augmentation
    #test_dataset_N = ImageFolder(root=args.test_folder, transform=ToTensor())
    #test_loaded_N = DataLoader(test_dataset_N, batch_size=args.batch, shuffle=True)


    # get covid data once I have it
    '''
    with open(trainfile) as f:
        trainfiles = f.readlines()
    with open(testfile) as f:
        testfiles = f.readlines()
    '''
    # balance dataset
    '''
    generator = BalanceCovidDataset(data_dir=datadir,
                                    csv_file=trainfile,
                                    covid_percent=covid_percent,
                                    class_weights=[1., 1., covid_weight])
    '''

    return train_loaded, val_loaded, test_loaded_A

# Train and evaluate
def train_model(model, train_loader, criterion, m, optimizer, scheduler, num_epochs=10,
                display_step=1):
    # out path
    outputPath = './output/'
    runID = name + '-lr' + str(learning_rate)
    runPath = outputPath + runID
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)


    print('Training started')
    losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()

        # Iterate over data batches
        for batch_idx, (inputs, y_batch) in enumerate(Bar(train_loader)):
            inputs = inputs.to(device)
            y_batch = y_batch.to(device)

            # load weights
            '''
            saver.restore(sess, os.path.join(args.weightspath, args.ckptname))'''
            # sample_weight = torch.empty(inputs.shape).uniform_(0, 1) # weights in covid given as argument
            # print(y_batch)
            sample_weight = np.take(class_weights, y_batch)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            loss = torch.mean(criterion(m(outputs), y_batch) * sample_weight)  # weights given as argument
            losses.append(loss)

            # backward
            loss.backward()
            optimizer.step()

        if epoch % display_step == 0:
            loss = torch.mean(criterion(m(outputs), y_batch) * sample_weight)
            print(loss.item())  # print some info - more detailed in covid
            _, preds = torch.max(outputs, 1)
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
            # call eval: TODO eval(sess, graph, testfiles, 'test')
            # save
            '''saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=False)'''
            # torch.save(model.state_dict(), weightspath) # saving the model in pytorch
            # print('Saving checkpoint at epoch {}'.format(epoch + 1))

        metric = 0
        scheduler.step(metric)


#######################

'''
class Training:
    def __init__(self, model,  epochs=10, learning_rate = 2e-5, bs = 8, weightspath = 'models/COVIDNet-CXR-Large', 
                 metaname = 'model.meta',ckptname = 'model-8485', trainfile = 'train_COVIDx2.txt', 
                 testfile = 'test_COVIDx2.txt', name = 'COVIDNet',datadir = 'data', covid_weight = 12, covid_percent = 0.3, class_weights = [1., 1., 6.] ):
        self.model = model
        self.num_epochs = epochs
        self.learning_rate = learning_rate
        self.bs = bs
        self.weightspath = weightspath
        self.metaname = metaname
        self.ckptname = ckptname
        self.trainfile = trainfile
        self.testfile = testfile
        self.name = name
        self.datadir = datadir
        self.covid_weight = covid_weight
        self.covid_percent = covid_percent
        self.class_weights = class_weights
'''


# added from args
learning_rate = 2e-5
num_epochs = 10
bs = 8
weightspath = 'models/COVIDNet-CXR-Large'
metaname = 'model.meta'
ckptname = 'model-8485'
trainfile = 'train_COVIDx2.txt'
testfile = 'test_COVIDx2.txt'
name = 'COVIDNet'
datadir = 'data'
covid_weight = 12
covid_percent= 0.3

class_weights=[1., 1., 6.]
display_step = 1

#### session starts
# get default graph
''' 
saver = .import_meta_graph(os.path.join(args.weightspath, args.metaname)) # import the pretrained weights
'''
# torch.save(model.state_dict(), weightspath) # saving the model in pytorch
# model = PyTorchModel()
# loaded_model = model.load_state_dict(torch.load(weightspath)) # load weights

# get var from loaded graph ??????
''' image_tensor = graph.get_tensor_by_name("input_1:0")
    labels_tensor = graph.get_tensor_by_name("dense_3_target:0")
    sample_weights = graph.get_tensor_by_name("dense_3_sample_weights:0")
    pred_tensor = graph.get_tensor_by_name("dense_3/MatMul:0")'''

# sample_weight = torch.empty((1,1)).uniform_(0, 1) # weights in covid given as argument
# labels_tensor = torch.empty((1,0)).uniform_(0, 1)
# pred_tensor = torch.empty((1,0)).uniform_(0, 1)



# Initialize the variables
'''
init = tf.global_variables_initializer()
equals tf.variables_initializer(var_list, name='init'), var_list = global_variables
'''
# pyTorch something like:
# init = nn.init() #?

# load weights
# loaded_model = model.load_state_dict(torch.load(weightspath)) # load weights

# save base model
# torch.save(model.state_dict(), weightspath) # saving the model in pytorch
# baseline eval
'''
  eval(sess, graph, testfiles, 'test')
'''

train_loader, val_loader, test_loader = loadData('data/train/', 'data/test/', 'data/test/', 8)
model = Net().to(device)

# optimzer + loss
m = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss()
# loss_op = torch.mean(criterion(m( pred_tensor), labels_tensor)* sample_weight)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, verbose=True)

train_model(model, train_loader, criterion, m, optimizer, scheduler, num_epochs=10,
                display_step=1)


