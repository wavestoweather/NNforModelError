
### IMPORTS
#from config_HR import *
from f_modRSW import step_forward_topog, time_step, make_grid
import time
import sys
import tools as t
import config_HR as HR
import numpy as np

"""
Imported quantities / information / tool functions

HR:
- int timesteps
- int Neq
- int k_max
- float L
- float H0
- float B_amplitude
- float B_0

t:
- string getMemberName()
(- {state, B, float B_max} load_ic(string input_dir, string member))
- B oro_rescale(B B_current, float amplitude_current, float amplitude_new)
"""

# file name convention
member = t.getMemberName(sys.argv[1])

# create HR grid
grid = make_grid(HR.HR, HR.L)
Kk = grid[0]
x = grid[1]
xc = grid[2]

# IC's and orography yr
U0, B, B_max = HR.ic(x, HR.HR, HR.Neq, HR.H0, HR.L, HR.k_max, HR.B_amplitude, HR.B_0)

uh_mean_init = np.mean(U0[1, :])

# state arrays
states = np.empty((HR.Neq, HR.HR, HR.timesteps))  # entire run
states[:, :, 0] = U0  # initial state
U = np.empty((HR.Neq, HR.HR))  # current state
U = U0

# integration loop parameters
t = 0
t_next = HR.dt
idx = 1
save = False

start = time.time()  # to measure computation time of integration
while idx < HR.timesteps:

    dt_dyn = time_step(U, Kk, HR.cfl_fc)  # dynamical time step
    t = t + dt_dyn

    # this is to ensure that states are saved at equidistant timesteps dt
    if t > t_next:
        dt_dyn -= (t - t_next)
        t = t_next
        save = True

    U = step_forward_topog(U, B, dt_dyn, HR.HR, Kk)

    if save:
        if HR.wind_mod == 'relaxation':
            uh_mean = np.mean(U[1, :])
            U[1,:] -= (uh_mean - uh_mean_init) / HR.t_relax
        states[..., idx] = U
        idx += 1
        t_next += HR.dt
        save = False

end = time.time()


print(
'\n',
'Run                       : ', member, '\n',
'The time I needed         : ', end - start, ' seconds', '\n',
'Initial domain mean wind  : ', uh_mean_init, '\n',
'Saving simulation data in : ', HR.output_dir, '\n',
)

np.save(str(HR.output_dir + '/states_' + member), states)
np.save(str(HR.output_dir + '/oro_' + member), B)
np.save(str(HR.output_dir + '/U0_' + member), U0)

