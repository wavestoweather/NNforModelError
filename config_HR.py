'''
Author: T. Kent  (amttk@leeds.ac.uk)
List of fixed parameters for model integration and EnKF.
Modified and extended by Raphael Kriegmair (LMU/MIM)
'''

import init_cond_modRSW as ICmodRSW

timesteps = 200000  # number of temporally equidistant states to save

Neq = 3          # number of equations in system (3 with topography, 4 with rotation)
L = 1.0          # length of domain (non-dim.)
bc = 'periodic'  # boundary conditions

# relaxation
wind_mod = 'relaxation'  # no_modification, relaxation
t_relax = 100.             # relaxation timescale in timesteps

HR = 800  # truth resolution

cfl_fc = 0.5  # Courant Friedrichs Lewy number for time stepping
cfl_tr = 0.5  # same

dt = 0.001          # TODO: describe

Ro = 'Inf'          # Rossby no. Ro ~ V0 / ( f * L0 )
Fr = 1.1            # Froude no.
g = Fr**(-2) 		# effective gravity, determined by scaling.
A = 0.1             # TODO: describe
V = 1.              # TODO: describe

# threshold heights
H0 = 1.0   # TODO: describe
Hc = 1.02  # TODO: describe
Hr = 1.05  # TODO: describe

# remaining parameters related to hr
beta = 0.2      # TODO: describe
alpha2 = 10     # TODO: describe
c2 = g * Hr     # TODO: describe
cc2 = 0.1 * c2  # the actual rain-to-potential conversion factor

# orography
ic = ICmodRSW.init_cond_topog_exp_rand  # TODO: describe
k_max = 100                             # TODO: describe
B_amplitude = 0.1                       # TODO: describe
B_offset = 0.                           # TODO: describe
B_0 = B_amplitude + B_offset            # TODO: describe

# experiment paths
output_dir = 'data/HR'
