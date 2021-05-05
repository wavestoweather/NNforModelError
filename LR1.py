
import numpy as np
import tools as t
import sys
import config_HR as HR
import config_LR1 as LR1

"""
Imported quantities / information / tool functions

HR:
- int timesteps
- int Neq
- int HR

LR1:
- string input_dir
- string output_dir
- int LR

t:
- string getMemberName()
- states, B coarse_grain()
"""

# file name convention
member = t.getMemberName(sys.argv[1])

print(' INFO: Loading HR data')
states_HR = np.load(LR1.input_dir + '/states_'+ member + '.npy') #np.load(LR1.input_dir + '/states_' + member + '.npy')
B_HR = np.load(LR1.input_dir + '/oro_' + member + '.npy')

print(' INFO: Defining LR grid')
states_LR = np.empty((HR.Neq, LR1.LR, HR.timesteps))
B_LR = np.empty((LR1.LR,))

print(' INFO: Performing coarse graining')
states_LR, B_LR = t.coarse_grain(states_HR, B_HR, HR.HR, LR1.LR)

print(' INFO: Saving results to ' + LR1.output_dir)
np.save(LR1.output_dir + '/states_' + member, states_LR)
np.save(LR1.output_dir + '/oro_' + member, B_LR)
