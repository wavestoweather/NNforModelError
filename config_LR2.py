'''
Parameters for LR forecast generation
'''

import numpy as np

numsteps_fc = 1  # number of timesteps for prediction
wind_mod = 'relaxation'  # no_modification, relaxation
t_relax = 100. # relaxation timescale in timesteps

# experiment paths
input_dir = 'data/LR1'
output_dir = 'data/LR2'

