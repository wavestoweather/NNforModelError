'''
Author: T. Kent  (amttk@leeds.ac.uk)
List of fixed parameters for model integration and EnKF.
Modified and extended by Raphael Kriegmair (LMU/MIM)
'''

import numpy as np

pureVariables = False  # use h, u, r instead of h, hu, hr
orography = True  # add orography as input channel
time_info = False  # add normalized time as input channel
# expand coordinate axis to achieve periodic padding via valid padding
periodic_convolution = True
excl_spinup = 100  # steps to exclude from beginning

# experiment paths
uni_work = '/project/meteo/work/Raphael.Kriegmair/hiwi/NN_paper/data'
uni_work = '/project/meteo/w2w/B6/Yvonne/NN_Paper_Code/data'

LR1_dir = 'data/LR1'
LR2_dir = 'data/LR2'
output_dir ='data/train'
