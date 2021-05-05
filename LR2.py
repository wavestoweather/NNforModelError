# -*- coding: utf-8 -*-
"""
Load coarse grained data and
integrate each state numsteps_fc timesteps forward.

@author: Raphael.Kriegmair
"""

import numpy as np
from f_modRSW import step_forward_topog, time_step, make_grid
import sys
import time
import tools as t
import config_HR as HR
import config_LR1 as LR1
import config_LR2 as LR2

"""
Imported quantities / information / tool functions

HR:
- float L
- float dt
- int timesteps
- float cfl_fc

LR1:
- int LR

LR2:
- string input_dir
- string output_dir
- int numsteps_fc
- bool wind_mod
- float t_relax

t:
- string getMemberName()
"""

# file name convention
member = t.getMemberName(sys.argv[1])

print(' INFO: Loading LR1 files')
states_LR1 = np.load(LR2.input_dir + '/states_' + member + '.npy')
B = np.load(LR2.input_dir + '/oro_' + member + '.npy')

uh_mean_init = np.mean(states_LR1[1, :, 0])

states_LR2 = np.zeros(states_LR1.shape)
Kk = make_grid(LR1.LR, HR.L)[0]  # LR grid cell length

print(' INFO: Integrating...')
timestep_fc = LR2.numsteps_fc * HR.dt
t = 0.
idx = 0
save = False

while idx < HR.timesteps:
    U = states_LR1[..., idx]

    while t < timestep_fc:
        dt_dyn = time_step(U, Kk, HR.cfl_fc)
        t = t + dt_dyn

        if t >= timestep_fc:
            dt_dyn = dt_dyn - (t - timestep_fc)
            save = True

        U = step_forward_topog(U, B, dt_dyn, LR1.LR, Kk)

        if save:
            if LR2.wind_mod == 'relaxation':
                uh_mean = np.mean(U[1, :])
                U[1, :] -= (uh_mean - uh_mean_init) / LR2.t_relax
            states_LR2[..., idx] = U
            save = False

    idx += 1
    t = 0.

print(' INFO: Saving results to ' + LR2.output_dir)
np.save(LR2.output_dir + '/states_' + member, states_LR2)
