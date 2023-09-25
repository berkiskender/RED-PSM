from pathlib import Path
import os
import itertools

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import radon, iradon
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage

import torch
from torch import nn as nn

import red_psm_models


def mask_fov_object(f, spatial_dim):
    """
    Applies the field-of-view mask for tomographic imaging to each frame of f.
    """
    for i,j in list(
        itertools.product(np.arange(spatial_dim), np.arange(spatial_dim))):
        if (i-spatial_dim//2)**2 + (j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
            f[i, j, :] = 0
    return f


def DecimalToBinary(n):
    return bin(n).replace("0b", "")


def bit_reversal(x, N):
    num_digit = 0
    while N // 2:
        N = N // 2
        num_digit += 1
    x = list(DecimalToBinary(x))
    while len(x) < num_digit:
        x = [0] + x
    x.reverse()
    return int("".join(str(n) for n in x), 2)


def obtain_projections(f, theta, P):
    """
    Computes the dynamic/undersampled projections of the time-varying object.
    """
    g = np.zeros([f.shape[0], P])
    for p in range(P):
        g[:, p] = radon(f[:, :, p], theta=[360 * theta[p] / (2 * np.pi)])[:, 0]
    return g


def obtain_all_projections(f, theta, P):
    """
    Computes the full set of projections of the time-varying object.
    """
    g = np.zeros([f.shape[0], len(theta), P])
    for p in range(P):
        g[:, :, p] = radon(f[:, :, p], theta=360 * theta / (2 * np.pi))
    return g


def compute_psnr(x_gt, x_rec):
    mse = np.mean((x_rec - x_gt) ** 2)
    return 20 * np.log10((np.max(x_gt) - np.min(x_gt)) / np.sqrt(mse))


def compute_mae(x_gt, x_rec):
    return np.mean(np.abs(x_rec - x_gt))


def compute_ssim(x_gt, x_rec):
    return ssim(x_gt, x_rec, data_range=x_rec.max() - x_rec.min())


def compute_hfen(f, f_est):
    hf_f = np.zeros(f.shape)
    log_filter = lambda x, y: ndimage.gaussian_laplace(x - y, sigma=1.5) 
    for t in range(f.shape[2]):
        hf_f[..., t] = log_filter(f[..., t], f_est[..., t])
    return np.linalg.norm(hf_f)


def generate_theta(P, ang_range=2 * np.pi, period=None):
    """
    Computes different view angle sampling schemes.
    """
    repeat_ang_sch = lambda x: np.tile(x, P//period)
    period = P if period is None else period
    theta_linear = repeat_ang_sch(np.linspace(0, ang_range, period, endpoint=False))
    theta_random = repeat_ang_sch(np.random.uniform(0, ang_range, size=[period]))
    theta_bit_reversal = repeat_ang_sch(np.array([(
        ang_range / period) * bit_reversal(p, period) for p in range(period)]))
    theta_golden_angle = repeat_ang_sch(np.array([((
        p * (111.25 / 360) * 2 * np.pi) % (2 * np.pi)) for p in range(period)]))
    return theta_linear, theta_random, theta_bit_reversal, theta_golden_angle


def construct_pi_symm_g(g_radon, g_radon_pi_symm, ang_range):
    """
    Returns measurements with pi-symmetric versions concatenated.
    """
    if ang_range == np.pi:
        # Concatenate pi-symm measurements as new rows
        g_radon_symm_long = np.zeros([2 * g_radon.shape[0], g_radon.shape[1]])
        g_radon_symm_long[:g_radon.shape[0], :] = g_radon
        g_radon_symm_long[g_radon.shape[0]:, :] = g_radon_pi_symm
    else:
        g_radon_symm_long = g_radon, g_radon
    return g_radon_symm_long


def generate_f_pol(f, P, spatial_dim, num_instances, theta_exp, obj_type,
                   period=None, path=None, save=True, add_path=None):
    """
    Computes full set of projections and FBP reconstruction for each time frame 
    of the object.
    """
    f_full_meas = np.zeros([spatial_dim, P, num_instances])
    f_true_recon = np.zeros(f.shape)
    for t in range(num_instances):
        f_full_meas[..., t] = radon(
            f[..., t * P // num_instances], theta=360 * theta_exp / (2 * np.pi))
        f_true_recon[..., t] = iradon(f_full_meas[..., t * P // num_instances],
                                       theta=360 * theta_exp / (2 * np.pi))
    if save:
        str_period = '' if period is None else '_' + str(period)
        np.save(path+'f_full_meas_%s%s_%d%s.npy' %(
            obj_type, add_path, P, str_period), f_full_meas)
        np.save(path+'f_true_recon_%s%s_%d%s.npy' %(
            obj_type, add_path, P, str_period), f_true_recon)
    return f_full_meas, f_true_recon


def add_meas_noise(g_radon, g_radon_pi_symm_long, noise_std, g_max, ang_range,
                   obj_type, P, period, path, save=False, add_path=None):
    """ 
    Adds gaussian noise with fixed std to the time-sequential projections.
    """
    g_radon_noisy = g_radon + np.random.normal(
        loc=0.0, scale=noise_std * g_max, size=g_radon.shape)
    g_radon_symm_long_noisy = np.zeros(g_radon_pi_symm_long.shape)
    g_radon_symm_long_noisy[:g_radon.shape[0]] = g_radon_noisy
    g_radon_symm_long_noisy[g_radon.shape[0]:] = g_radon_pi_symm_long[
        g_radon.shape[0]:] + np.flipud(g_radon_noisy - g_radon)
    if save:
        if ang_range == np.pi:
            str_period = '' if period is None else '_' + str(period)
            np.save(
                path + 'g_radon_symm_long_noisy_%s%s_%d%s_noise_std_%.2e.npy' %(
                obj_type, add_path, P, str_period, noise_std),
                g_radon_symm_long_noisy)
    return g_radon_noisy, g_radon_symm_long_noisy


class patchifier(torch.nn.Module):
    """
    Returns the spatially patchified version of the input object at each frame.
    """
    def __init__(self, patchSize, patchStride, spatial_dim, bs):
        super(patchifier, self).__init__()
        
        self.patchSize, self.patchStride, self.bs = patchSize, patchStride, bs
        self.fold_params = dict(
            kernel_size=[patchSize, patchSize], stride=patchStride)
        self.fold = nn.Fold(
            output_size=[spatial_dim, spatial_dim], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)
        
    def merge(self, patches):
        patches = patches.contiguous().view(
            self.bs, patches.shape[0]//self.bs,
            self.patchSize*self.patchSize).permute(0,2,1)
        x = self.fold(patches)
        return x/(self.patchSize**2/self.patchStride**2)
    
    def forward(self, x):
        patches = self.unfold(x).permute(0,2,1)
        patches = patches.contiguous().view(
            self.bs*patches.shape[1], 1, self.patchSize, self.patchSize)
        return patches 


def denoising_network_loader(train_type, denoiser_type, pSize, pStride, 
                             num_layers, spatial_dim, obj_type, 
                             num_channels, noise_est_type='direct', 
                             epochs=500):
    '''
    Loads RED denoiser with the provided specifications.

    Parameters:
    ----------
    train_type (str): The type of denoising network.
    denoiser_type (str): The type of denoising model. 
        Options: ['full_img', 'patch_based', and 'patch_based_patchloss']
    pSize (int): The size of the patch. Only applicable for patch-based models.
    pStride (int): Patch extraction stride. Only applicable for patch-based models.
    num_layers (int): The number of layers in the denoising network.
    spatial_dim (int): The spatial dimension of f.
    obj_type (str): The type of object being denoised.
    num_channels (int): The number of channels for each layer of denoiser.
    noise_est_type (str, optional): Denoising type. 
        Options: ['direct', 'residual']
    epochs (int, optional): The number of pre-training epochs. Default is 500.
    
    Output:
    -------
    model_dncnn (torch.nn.Module): The loaded and configured deep RED denoiser.
    patchifier_red (Patchifier or None): A patchifier for patch-based denoising. 
        None for full image denoisers.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filterSize = 3
    if train_type == 'dncnn':
        if denoiser_type == 'full_img':
            model_name = 'dncnn'
            model_path = os.path.join(
                'data/denoiser',
                model_name + '_model_%s_%s_epochs_%d_num_layers_%d_num_ch_%d.pt' %(
                obj_type, noise_est_type, epochs, num_layers, num_channels))
            model_dncnn = red_psm_models.dncnn(
                num_layers, num_channels, filterSize, noise_est_type).cuda()
            patchifier_red = None
        elif denoiser_type == 'patch_based_patchloss':
            model_name = 'dncnn_patchbased_patchloss'
            model_path = os.path.join(
                'data/denoiser',
                model_name + '_model_%s_%s_num_layers_%d_patch_size_%d_patch_stride_%d_num_ch_%d_epochs_%d.pt' %(
                    obj_type, noise_est_type, num_layers, pSize, pStride,
                    num_channels, epochs))
            model_dncnn = red_psm_models.dncnnPatchBased_patchLoss(
                num_layers, num_channels, filterSize, noise_est_type).cuda()
            patchifier_red = patchifier(pSize, pStride, spatial_dim, 1)
        else:
            raise NotImplementedError
        
    model_dncnn = torch.load(model_path)
    model_dncnn.eval()
    for k, v in model_dncnn.named_parameters():
        v.requires_grad = False
    model_dncnn = model_dncnn.to(device)
    number_parameters = sum(map(lambda x: x.numel(), model_dncnn.parameters()))
    print('DnCNN %s Model path: {%s} Params number: %d'%(
        denoiser_type, model_path, number_parameters))
    return model_dncnn, patchifier_red


def load_radon_op(pi_symm, spatial_dim, P, period=None, path=None):
    '''
    Loads the differentiable forward operator (Radon transform).

    Parameters:
    -----------
    pi_symm (bool): Indicates if the Radon operator incorporates pi-symmetry.
    spatial_dim (int): The spatial dimension of f.
    P (int): The number of measurements.
    period (int, optional): The number of repeated unique view angles. 
        If not provided, it defaults to P.
    path (str, optional): Forward operator path.

    Output:
    -------
    R (numpy.ndarray): The Radon operator matrix.
    R_cuda (torch.Tensor): The Radon operator matrix as a CUDA tensor for GPU.
    '''
    if period == None:
        period = P
    if pi_symm:
        R = np.load(
            path + 'A_radon_spatial_dim_%d_P_%d_bit_reversal_pi_symm.npy' %(
                spatial_dim, period))
    else:
        R = np.load(
            path + 'A_radon_spatial_dim_%d_P_%d_bit_reversal.npy' %(
                spatial_dim, period))
    R = np.tile(np.array(R), (P//period, 1, 1))
    R_cuda = torch.cuda.FloatTensor(R)
    return R, R_cuda


def load_f(obj_type, motion, spatial_dim, P):
    if obj_type in ['walnut']:
        f = np.load(
            'data/true_objects/%s/f_%s_%s_spatial_dim_%d_P_%d.npy' %(
                obj_type, obj_type, motion, spatial_dim, P))
    else:
        raise NotImplementedError
    return f
