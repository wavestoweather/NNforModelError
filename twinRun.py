# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:03:03 2019

Load a trained network and create a twin run initialized with model truth state.

@author: Raphael.Kriegmair
"""

import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import sys
from f_modRSW import make_grid
import tools as t
import config_HR as HR
import config_LR1 as LR1
import config_LR2 as LR2
import config_training_data as TD
import config_kerasConv1D as kerasConv1D
import config_twinRun as TR
import os

"""
Imported quantities / information / tool functions

HR:
- int timesteps
- int Neq
- float L
- float dt

LR1:
- int LR

LR2:
- int numsteps_fc
- string wind_mod
- float t_relax

TD:
- int excl_spinup
- bool orography
- bool time_info
- bool periodic_convolution

kerasConv1D:
- float train_ratio
- int kernel_size
- int hiddenLayers

TR:
- float mass_weight
- string network_dir
- string LR1_dir
- string train_dir
- string output_dir
- int numsteps_per_hour
- int iterations

t:
- string getMemberName()
- state forward_topog_timestep(state current, int HR, B, int LR, float Kk)
- state prepare(state, float[] x_mu, float[] x_sigma, int grid_extension)
- state denormalize(state corr, float[] y_mu, float[] y_sigma)
(- h_state norm_mass(h_state, B))
"""

path = TR.output_dir+'/weight'+str(sys.argv[4])+'/epoch'+str(sys.argv[3])
if not os.path.exists(path):
    os.makedirs(path)

shift = int(sys.argv[2])*144*2  # in hours
member = t.getMemberName(sys.argv[1])

output_name = '/' + member
output_name += '_shift' + "%.1f" % shift

if os.path.isfile(path + output_name + '_states_corr.npy'):
    print('file already exists')
    sys.exit()
    
print("INFO: Reading data ...")

# Custom loss function
def test_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    mass_cons_violation = sys.argv[5] * K.mean( K.square(K.mean(y_pred[..., 0],axis=-1) ))  #yr
    return mse + mass_cons_violation

# Network
model = keras.models.load_model(TR.network_dir + '/weight'+str(sys.argv[4])+'/NN_' + member + '_'
                                + str(sys.argv[3]),
                                custom_objects={'test_loss': test_loss})

# time series
# (channel, coordinate, time)
mtr = np.load(TR.LR1_dir + '/states_' + member + '.npy')[..., TD.excl_spinup:]  # exclude spinup
B = np.load(TR.LR1_dir + '/oro_' + member + '.npy')  # original orography

if TD.orography:
    B_norm = np.load(TR.train_dir + '/B_norm_' + member + '.npy')
if TD.time_info:
    time_array = np.load(TR.train_dir + '/time_info_' + member + '.npy')


# normalization quantities
x_mu = np.load(TR.train_dir + '/x_mu_' + member + '.npy')
x_sigma = np.load(TR.train_dir + '/x_sigma_' + member + '.npy')
y_mu = np.load(TR.train_dir + '/y_mu_' + member + '.npy')
y_sigma = np.load(TR.train_dir + '/y_sigma_' + member + '.npy')


#   A R R A Y S   ###

# time index should be on first axis:
# --> (time, coordinate, channel)
mtr = np.swapaxes(mtr, 0, 2)

x_mu = np.swapaxes(x_mu, 0, 2)
x_sigma = np.swapaxes(x_sigma, 0, 2)
y_mu = np.swapaxes(y_mu, 0, 2)
y_sigma = np.swapaxes(y_sigma, 0, 2)

states_corr = np.zeros((TR.iterations,mtr.shape[1],mtr.shape[2]))  # CORRECTED model run
states = np.zeros(states_corr.shape)  # NON-corrected model run
corrections = np.zeros(states_corr.shape)  # ANN corrections for analysis
#   P R E D I C T I O N S   ###

print("INFO: Generating predictions")

num_channels = HR.Neq + int(TD.orography) + int(TD.time_info)
grid_extension = 0
if TD.periodic_convolution:
    # NOTE: in addition to the hidden layers, we have convolutional input and output layers!
    grid_extension = int((kerasConv1D.kernel_size - 1) * (kerasConv1D.hiddenLayers + 2) / 2)

# holds current ANN input, ANN expects additional time index (first axis)
NN_input = np.zeros((1, LR1.LR + 2 * grid_extension, num_channels))

if TD.orography and not TD.time_info:
    NN_input[..., 3] = B_norm
elif TD.time_info and not TD.orography:
    NN_input[..., 3] = time_array[:, 0]
elif TD.orography and TD.time_info:
    NN_input[..., 3] = B_norm
    NN_input[..., 4] = time_array[:, 0]


# initialize with model truth
states_corr[0, ...] = mtr[shift, ...]
states[0, ...] = mtr[shift, ...]

# compute initial mean wind
uh_mean_init = np.mean(states[0, :, 1])
Kk = make_grid(LR1.LR, HR.L)[0]

for idx in range(0, TR.iterations - 1):

    # feed current states into model
    states[idx+1, ...] = t.forward_topog_timestep(states[idx, ...], HR.dt, B, LR1.LR, Kk)
    states_corr[idx+1, ...] = t.forward_topog_timestep(states_corr[idx, ...], HR.dt, B, LR1.LR, Kk)

    if LR2.wind_mod == 'relaxation':
        states_uh_mean = np.mean(states[idx+1, :, 1])
        states_corr_uh_mean = np.mean(states_corr[idx+1, :, 1])
        states[idx+1, :, 1] -= (states_uh_mean - uh_mean_init) / LR2.t_relax
        states_corr[idx+1, :, 1] -= (states_corr_uh_mean - uh_mean_init) / LR2.t_relax

    # prepare current state for neural network
    NN_input[0, :, :3] = t.prepare(states_corr[idx, ...], x_mu, x_sigma, grid_extension)
    if TD.time_info and not TD.orography:
        NN_input[0, :, 3] = time_array[:, idx]
    elif TD.time_info and TD.orography:
        NN_input[0, :, 4] = time_array[:, idx]

    # obtain correction from neural network
    corr = model.predict(NN_input)[0, ...]
    corr = t.denormalize(corr, y_mu, y_sigma)
    
    # add network correction to model forecast
    states_corr[idx+1, ...] += corr[0, ...]
    corrections[idx+1, ...] = corr[0, ...]

    


states_corr = np.swapaxes(states_corr, 0, 2)
states = np.swapaxes(states, 0, 2)
corrections = np.swapaxes(corrections, 0, 2)

np.save(path + output_name + '_states_corr', states_corr)
np.save(path + output_name + '_states', states)
np.save(path + output_name + '_corrections', corrections)
