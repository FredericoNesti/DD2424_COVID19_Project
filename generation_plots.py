import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
os.chdir('/home/firedragon/Desktop/PROJECTS/COVID19/Results/Graphs')

Lower_Bound_total_train = np.load('Lower_Bound_total_train.npy')
Lower_Bound_encoder_train = np.load('Lower_Bound_encoder_train.npy')
Lower_Bound_decoder_train = np.load('Lower_Bound_decoder_train.npy')

Lower_Bound_total_val = np.load('Lower_Bound_total_val.npy')
Lower_Bound_encoder_val = np.load('Lower_Bound_encoder_val.npy')
Lower_Bound_decoder_val = np.load('Lower_Bound_decoder_val.npy')

Acc_ssim_train = np.load('Acc_ssim_train.npy')
Acc_ssim_val = np.load('Acc_ssim_val.npy')

num_epochs = Acc_ssim_val.shape[0]

print(num_epochs)

Epochs = [epoch for epoch in range(1, 100 + 1)]

# outputs from training
plt.figure()
plt.plot(Epochs, Lower_Bound_total_train, label='Total')
plt.plot(Epochs, Lower_Bound_encoder_train, label='Encoder')
plt.plot(Epochs, Lower_Bound_decoder_train, label='Decoder')
plt.yscale('log')
plt.legend()
plt.title('Lower Bound detail: Train set')
plt.xlabel('Epochs')
plt.show()
#plt.savefig(args.results_directory + '/Graphs/CNNVAEloss_train.jpg')

# outputs from validation
plt.figure()
plt.plot(Epochs, Lower_Bound_total_val, label='Total')
plt.plot(Epochs, Lower_Bound_encoder_val, label='Encoder')
plt.plot(Epochs, Lower_Bound_decoder_val, label='Decoder')
plt.yscale('log')
plt.legend()
plt.title('Lower Bound detail: Validation set')
plt.xlabel('Epochs')
plt.show()
#plt.savefig(args.results_directory + '/Graphs/CNNVAEloss_validation.jpg')

#
plt.figure()
plt.plot(Epochs, Lower_Bound_total_train, label='Train')
plt.plot(Epochs, Lower_Bound_total_val, label='Validation')
plt.yscale('log')
plt.legend()
plt.title('Lower Bound')
plt.xlabel('Epochs')
plt.show()
#plt.savefig(args.results_directory + '/Graphs/CNNVAEloss_train_plus_val.jpg')

plt.figure()
plt.plot(Epochs, Acc_ssim_train, label='Train')
plt.plot(Epochs, Acc_ssim_val, label='Validation')
plt.yscale('linear')
plt.legend()
plt.title('Structural Similarity Index')
plt.xlabel('Epochs')
plt.show()
#plt.savefig(args.results_directory + '/Graphs/CNNVAEssim.jpg')
