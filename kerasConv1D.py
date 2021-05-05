    # -*- coding: utf-8 -*-
"""
Train convolutional neural network.
Input: LR1[t]
Output: LR2[t] - LR1[t+1] (LR2 states are one time step ahead of LR1.)

keras source: https://keras.io/layers/convolutional/
"""

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Activation
from tensorflow.keras.metrics import MeanSquaredError
import numpy as np
import sys
import matplotlib as mpl
mpl.use('Agg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import tools as t
import config_HR as HR
import config_LR2 as LR2
import config_training_data as TD
import config_kerasConv1D as kerasConv1D
import pickle

def load(fn):
    """Loads the pickle files that the nsw model outputs"""
    with open(fn, 'rb') as f:
        return pickle.load(f, encoding='latin1')   # Not quite sure why this is necessary.

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
member = t.getMemberName(sys.argv[1])

print("INFO: Reading training samples ...")
# (channel, coordinate, time)
x = np.load(kerasConv1D.train_dir + '/x_samples_' + member + '.npy')
y = np.load(kerasConv1D.train_dir + '/y_samples_' + member + '.npy')

# Loss function
def test_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred),1)  #K.mean(K.sqrt(K.mean(K.square(y_true - y_pred),axis=1)),axis=-1) 
    mass_cons_violation = kerasConv1D.mass_weight * K.mean( K.square(K.mean(y_pred[..., 0],axis=-1) )) #kerasConv1D.mass_weight * K.mean( K.sqrt(K.square( K.mean(y_pred[..., 0],1)) ))   #yr
    return mse + mass_cons_violation

num_samples = HR.timesteps - LR2.numsteps_fc
num_train_samples = int(num_samples * kerasConv1D.train_ratio)
num_val_samples = num_samples - num_train_samples
print('number of training samples ',num_train_samples)
train_x, val_x = x[..., np.arange(0,x.shape[-1],2)], x[..., np.arange(1,x.shape[-1],2)]
train_y, val_y = y[...,  np.arange(0,x.shape[-1],2)], y[...,  np.arange(1,x.shape[-1],2)]

# time/sample index must be on first axis: (timestep/sample, coordinate, variable/channel)
train_x = np.swapaxes(train_x, 0, 2)
train_y = np.swapaxes(train_y, 0, 2)
val_x = np.swapaxes(val_x, 0, 2)
val_y = np.swapaxes(val_y, 0, 2)

print("INFO: Initializing model")

# NOTE: input second axis is where convolution is carried out.
# In my case, its the coordinate axis.
# The last (here: third) one must be "channels".
model = Sequential()
model.add(Conv1D(filters=kerasConv1D.filters,
                 kernel_size=kerasConv1D.kernel_size,
                 padding=kerasConv1D.padding,
                 use_bias=True,
                 input_shape=train_x[0, ...].shape,
                 activation=kerasConv1D.activation,
                 data_format='channels_last'))

for layers in range(kerasConv1D.hiddenLayers):
    model.add(Conv1D(filters=kerasConv1D.filters,
                     kernel_size=kerasConv1D.kernel_size,
                     padding=kerasConv1D.padding,
                     use_bias=True,
                     activation=kerasConv1D.activation,
                     data_format='channels_last'))

model.add(Conv1D(filters=3,
                 kernel_size=kerasConv1D.kernel_size,
                 padding=kerasConv1D.padding,
                 use_bias=True,
                 activation='linear',
                 data_format='channels_last'))

from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(kerasConv1D.lr), # 'adam'
              loss=test_loss,
              metrics='mse')

print("INFO: Training")
my_callbacks = [ModelCheckpoint(filepath= kerasConv1D.network_dir + '/NN_' + member + '_{epoch:02d}',
                monitor='val_loss', verbose=0, save_best_only=False,
                save_weights_only=False, mode='auto', period=1),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=kerasConv1D.patience,
                verbose=0, mode='auto', baseline=None, restore_best_weights=True)]

hist = model.fit(train_x, train_y,
                 epochs=kerasConv1D.epochs,
                 validation_data=(val_x, val_y),
                 shuffle=True,
                 batch_size = kerasConv1D.batch_size,
                 callbacks = my_callbacks)

# save history objects and model
save_obj(hist.history, str(kerasConv1D.network_dir + '/history_' + member) )
model.save(kerasConv1D.network_dir + '/NN_' + member)
