import os
import torch
import numpy as np
from visdom import Visdom

def save_model(args_dict, state):
    """
    Saves model
    """
    directory = args_dict.dir_model
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args_dict.mode == 'calibration':
        filename = directory + "calibrated_" + args_dict.model + '_best_model.pth.tar'
    else:
        filename = directory + args_dict.model + '_best_model.pth.tar'

    torch.save(state, filename)

def resume(args_dict, model, optimizer):
    """
    Continue training from a checkpoint
    :return: args_dict, model, optimizer from the checkppoint
    """
    best_sensit = -float('Inf')
    args_dict.start_epoch = 0
    if args_dict.resume:
        model_path = args_dict.dir_model + args_dict.model + '_best_model.pth.tar'
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(args_dict.resume))
            checkpoint = torch.load(model_path, map_location=torch.device(args_dict.device))
            args_dict.start_epoch = checkpoint['epoch']
            best_sensit = checkpoint['best_sensit']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args_dict.model, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))
            best_sensit = -float('Inf')

    return best_sensit, model, optimizer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
