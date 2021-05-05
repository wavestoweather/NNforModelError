# -*- coding: utf-8 -*-

lead = 48  # hours
numsteps_per_hour = 144  # TODO: still hardcoded, find rule to compute
iterations = lead * numsteps_per_hour  # timesteps

# experiment paths
train_dir = 'data/train'
network_dir = 'data/NN'
LR1_dir = 'data/LR1'
output_dir = 'data/twinRun' 
