'''
Author: T. Kent  (amttk@leeds.ac.uk)
List of fixed parameters for model integration and EnKF.
Modified and extended by Raphael Kriegmair (LMU/MIM)
'''
#import tools_keras as t_keras

train_ratio = 0.5
val_ratio = 1. - train_ratio

activation = 'relu'  # activations: relu, sigmoid, ...
hiddenLayers = 5 #10
filters = 32 #60
kernel_size = 3
epochs = 1000
batch_size = 256 #32
lr = 0.001  # learning rate
padding = 'valid'  # 'same', 'valid' ...
mass_weight = 0  # weight for mass conservation violation loss
patience = 1000

# experiment paths
train_dir = 'data/train'
network_dir = 'data/NN/weight'+str(mass_weight)
