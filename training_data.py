# -*- coding: utf-8 -*-
"""
Preprocessing of LR1 and LR2 data.
Produces normalized data formatted for network training.

@author: Raphael.Kriegmair
"""

import numpy as np
import sys
import tools as t
import config_HR as HR
import config_LR1 as LR1
import config_LR2 as LR2
import config_training_data as TD
import config_kerasConv1D as kerasConv1D

"""
Imported quantities / information / tool functions

HR:
- int Neq

LR1:
- int LR

LR2:
- int numsteps_fc

TD:
- string LR1_dir
- string LR2_dir
- string output_dir
- bool pureVariables
- bool orography
- bool time_info
- bool periodic_convolution
- int excl_spinup

t:
- string getMemberName()
- {states normalized_data, float[] mu, float[] sigma} get_norm(states data)
"""

# file name convention
member = t.getMemberName(sys.argv[1])

print("INFO: Reading time series ...")

# (variable/channel, coordinate, time)
mtr = np.load(TD.LR1_dir + '/states_' + member + '.npy')  # model truth LR1
fc = np.load(TD.LR2_dir + '/states_' + member + '.npy')  # LR prediction LR2
B = np.load(TD.LR1_dir + '/oro_' + member + '.npy')  # LR orography

print("INFO: Preprocessing data ...")

# convert (h, hu, hr) to (h, u, r)
if TD.pureVariables:
    print('  Using PURE variables')
    mtr[1:, ...] /= mtr[0, ...]
    fc[1:, ...] /= fc[0, ...]

# define input and target s.t. pairs share indices
diff = mtr[..., LR2.numsteps_fc:] - fc[..., :-LR2.numsteps_fc]  # target
mtr = mtr[..., :-LR2.numsteps_fc]  # no target exists for last 'numsteps_fc' model truth states
#mtr = fc[..., :-LR2.numsteps_fc] #yr 

# exclude spinup phase
diff = diff[..., TD.excl_spinup:]
mtr = mtr[..., TD.excl_spinup:]

print('  Normalization...')

x_norm, x_mu, x_sigma = t.get_norm(mtr)
y_norm, y_mu, y_sigma = t.get_norm(diff)
B_mu = np.mean(B)
B_sigma = np.std(B)
B_norm = (B - B_mu) / B_sigma

x_extd = x_norm
num_samples = x_norm.shape[2]
if TD.orography:
    print('  Adding input channel: orography')
    # create B-array
    B_array = np.repeat(B_norm[:, np.newaxis], num_samples, axis=1)
    # reshape input
    extdShape = (HR.Neq+1, LR1.LR, num_samples)
    x_extd = np.zeros(extdShape)
    x_extd[:HR.Neq, ...] = x_norm
    x_extd[HR.Neq, ...] = B_array

if TD.time_info:
    print('  Adding input channel: timestep information')
    # create timestep-array
    time_array = np.zeros((LR1.LR, num_samples))
    for t in range(num_samples):
        time_array[..., t] = t
    # normalize
    time_array /= num_samples
    # reshape input
    if TD.orography:
        extdShape = (HR.Neq+2, LR1.LR, num_samples)
        x_extd = np.zeros(extdShape)
        x_extd[:HR.Neq, ...] = x_norm
        x_extd[HR.Neq, ...] = B_array
        x_extd[HR.Neq+1, ...] = time_array
    else:
        extdShape = (HR.Neq+1, LR1.LR, num_samples)
        x_extd = np.zeros(extdShape)
        x_extd[:HR.Neq, ...] = x_norm
        x_extd[HR.Neq, ...] = time_array

if TD.periodic_convolution:
    x_shape = x_extd.shape
    # NOTE: in addition to hidden layers, we have convolutional input and output layers!
    grid_extension = int((kerasConv1D.kernel_size - 1) * (kerasConv1D.hiddenLayers + 2) / 2)
    extd_range = x_shape[1] + 2 * grid_extension
    extdShape = (x_shape[0], extd_range, x_shape[2])
    result = np.zeros(extdShape)
    result[:, :grid_extension, :] = x_extd[:, -grid_extension:, :]
    result[:, grid_extension:grid_extension+x_shape[1], :] = x_extd
    result[:, -grid_extension:, :] = x_extd[:, :grid_extension, :]
    x_extd = result

print('INFO: Saving results to ' + TD.output_dir)

np.save(TD.output_dir + '/x_samples_' + member, x_extd)
np.save(TD.output_dir + '/y_samples_' + member, y_norm)
np.save(TD.output_dir + '/x_mu_' + member, x_mu)
np.save(TD.output_dir + '/x_sigma_' + member, x_sigma)
np.save(TD.output_dir + '/y_mu_' + member, y_mu)
np.save(TD.output_dir + '/y_sigma_' + member, y_sigma)

if TD.orography:
    np.save(TD.output_dir + '/B_norm_' + member, x_extd[HR.Neq, :, 0])
    np.save(TD.output_dir + '/B_mu_' + member, B_mu)
    np.save(TD.output_dir + '/B_sigma_' + member, B_sigma)

# TODO: this probably doesnt make too much sense,
# adjust when using timestep info
if TD.time_info:
    np.save(TD.output_dir + '/time_info_' + member, time_array)
