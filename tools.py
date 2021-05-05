
from f_modRSW import step_forward_topog, time_step, make_grid
from copy import deepcopy
import numpy as np
import config_HR as HR
import config_LR1 as LR1
import config_training_data as TD
import config_analysis as analysis


"""
load_ic
fss
getMemberName
oro_rescale
coarse_grain
forward_topog_timestep
prepare
expand
get_norm
normalize
denormalize
diff_comp2pure
norm_mass
bin_data
"""
def moving_average(x,window):
    n = len(x)
    y = np.zeros(n-window+1)
    for i in range(len(y)):
        
        y[i] = sum(x[i:i+window])/window
    return y

def optimal_epoch(data, patience):
    idx = 0
    val = data[1, 0]
    while idx + patience < data.shape[1]:
        min_idx = np.argmin(data[1, idx:idx+patience])
        min_val = data[1, min_idx]
        if min_val < val:
            idx = min_idx
            val = min_val
        else:
            return idx


def load_ic(input_dir, member):
    U0 = np.load(input_dir + '/U0_' + member + '.npy')
    B = np.load(input_dir + '/oro_' + member + '.npy')
    B_max = np.load(input_dir + '/info_' + member + '.npy')[0]

    return U0, B, B_max


def fss(_truth, _fc, B, threshold, size):
    truth_mask = np.zeros(_truth.shape)
    fc_mask = np.zeros(_truth.shape)
    result = np.zeros(_truth.shape)
    truth = deepcopy(_truth)
    fc = deepcopy(_fc)
    truth[0, ...] += np.repeat(B[:, np.newaxis], truth.shape[2], axis=1)
    fc[0, ...] += np.repeat(B[:, np.newaxis], truth.shape[2], axis=1)
    truth_mask[0, ...] = truth[0, ...] > threshold[0]
    truth_mask[1, ...] = truth[1, ...] > threshold[1]
    truth_mask[2, ...] = truth[2, ...] > threshold[2]
    fc_mask[0, ...] = fc[0, ...] > threshold[0]
    fc_mask[1, ...] = fc[1, ...] > threshold[1]
    fc_mask[2, ...] = fc[2, ...] > threshold[2]
    truth_ext = expand(truth_mask, size)
    fc_ext = expand(fc_mask, size)

    for i in range(LR1.LR):
        count_truth = np.sum(truth_ext[:, i:i+2*size, :], axis=1)
        count_fc = np.sum(fc_ext[:, i:i+2*size, :], axis=1)
        result[:, i, :] = np.abs(count_truth - count_fc)

    return result


# name files s.t. they are shown in the right order in folder
def getMemberName(member):
    if int(member) < 100:
        if int(member) < 10:
            member = 'member_00' + str(member)
        else:
            member = 'member_0' + str(member)
    else:
        member = 'member_' + str(member)

    return member



def oro_rescale(oro, amp_old, amp_new):
    return (oro - amp_old) * amp_new / amp_old + amp_new


# TODO: HR loop does not need to go over all cells each time, reformulate.
def coarse_grain(U_HR, B_HR, hr, lr):

    timesteps = U_HR.shape[2]
    U_LR = np.zeros((HR.Neq, lr, timesteps))  # states result array
    B_LR = np.zeros((lr,))  # orography result array
    grid_HR = make_grid(hr, HR.L)
    grid_LR = make_grid(lr, HR.L)
    # edge coordinates
    x_HR = grid_HR[1]
    x_LR = grid_LR[1]
    # cell lengths
    dx_HR = grid_HR[0]
    dx_LR = grid_LR[0]

    for i in range(lr):
        # result values for current LR cell
        U_vals = np.zeros((HR.Neq, timesteps))
        B_val = 0.
        # find 'overlapping' HR grid cells on left/right LR cell edges
        # determine 'overlap fractions'
        x_LR_left = x_LR[i]
        x_LR_right = x_LR[i+1]
        left_overlap_frac = 0.
        right_overlap_frac = 0.
        idx_HR_left = 0
        idx_HR_right = 0
        for j in range(hr):
            if x_HR[j] <= x_LR_left and x_HR[j+1] >= x_LR_left:
                left_overlap = x_HR[j+1] - x_LR_left
                idx_HR_left = j
            if x_HR[j] <= x_LR_right and x_HR[j+1] >= x_LR_right:
                right_overlap = x_LR_right - x_HR[j]
                idx_HR_right = j
        # sum up contributions
        for j in range(idx_HR_left+1, idx_HR_right):
            U_vals += U_HR[:, j, :] * dx_HR
            B_val += B_HR[j] * dx_HR
        U_vals += U_HR[:, idx_HR_left, :] * left_overlap
        U_vals += U_HR[:, idx_HR_right, :] * right_overlap
        B_val += B_HR[idx_HR_left] * left_overlap
        B_val += B_HR[idx_HR_right] * right_overlap

        #num_cells = idx_HR_right + 1 - idx_HR_left  # number of contributions
        # make average
        U_vals /= dx_LR
        B_val /= dx_LR

        U_LR[:, i, :] = U_vals
        B_LR[i] = B_val

    return U_LR, B_LR


# Integrates state forward in time for a given timestep length "T"
def forward_topog_timestep(U, T, B, res_hr, Kk):
    U = np.swapaxes(U, 0, 1)
    save = False
    t = 0.
    while t < T:

        dt = time_step(U, Kk, HR.cfl_fc)
        t = t + dt

        if t >= T:
            dt = dt - (t - T)
            save = True

        U = step_forward_topog(U, B, dt, res_hr, Kk)

        if save:
            return np.swapaxes(U, 0, 1)


# prepare current state for neural network
# state already includes orography
# rename argument since this function would change its input!???
def prepare(state, LR1_mean, LR1_sigma, grid_extension, pureVars=False):
    state_copy = deepcopy(state)
    if pureVars:
        state_copy[:, 1:2] /= state_copy[:, 0]

    state_copy = normalize(state_copy, LR1_mean, LR1_sigma)
    if TD.periodic_convolution:
        state_copy = expand(state_copy, grid_extension)

    return state_copy


def expand(state, grid_extension):
    x_shape = state.shape
    # NOTE: in addition to hidden layers, we have convolutional input and output layers!
    extd_range = x_shape[1] + 2 * grid_extension
    extdShape = (x_shape[0], extd_range, x_shape[2])
    result = np.zeros(extdShape)
    result[:, :grid_extension, :] = state[:, -grid_extension:, :]
    result[:, grid_extension:grid_extension+x_shape[1], :] = state
    result[:, -grid_extension:, :] = state[:, :grid_extension, :]
    return result


def get_norm(x):
    mu = np.mean(x, axis=(1, 2), keepdims=True)
    sigma = np.std(x, axis=(1, 2), keepdims=True)
    return (x - mu) / sigma, mu, sigma


def normalize(x, mu=None, sigma=None):
    return (x - mu) / sigma


def denormalize(x, mu, sigma):
    return x * sigma + mu


# convert network correction
# from (delta h, delta hu, delta hr)
# to (delta h, delta u, delta r)
# input shapes: (..., var)
# returns pure corrections
def diff_comp2pure(corrected, corrections):

    comp_corrected = deepcopy(corrected)
    comp_corrections = deepcopy(corrections)

    pure_corrected = np.zeros(comp_corrected.shape)
    pure_uncorrected = np.zeros(comp_corrected.shape)

    #import pdb
    #pdb.set_trace()

    pure_corrected[0, ...] = comp_corrected[0, ...]
    pure_corrected[1, ...] = comp_corrected[1, ...] / comp_corrected[0, ...]
    pure_corrected[2, ...] = comp_corrected[2, ...] / comp_corrected[0, ...]

    comp_uncorrected = comp_corrected - comp_corrections
    pure_uncorrected[0, ...] = comp_uncorrected[0, ...]
    pure_uncorrected[1, ...] = comp_uncorrected[1, ...] / comp_uncorrected[0, ...]
    pure_uncorrected[2, ...] = comp_uncorrected[2, ...] / comp_uncorrected[0, ...]

    return pure_corrected - pure_uncorrected


# normalize fluid height field to initial total fluid mass.
# input: current height field h, orography b
# returns normalized fluid height field.
def norm_mass(h, b):

    init_mass = np.sum(H0-b)
    current_mass = np.sum(h)
    N = init_mass / current_mass

    return N * h



def bin_data(data_, res):
    data = np.asarray(data_)
    max = np.max(data)
    min = np.min(data)
    width = (max - min) / res
    result = np.zeros(res)
    for d in data:
        for i in range(0, res):
            if ((d >= min + i*width) and (d < min + (i+1)*width)):
                result[i] += 1

    return result
