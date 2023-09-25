import itertools

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torch_cubic_spline_interp


def generate_psi_from_z(K, P, temporal_fcts, z_dim, temporal_basis, D_t, P_t, 
                        gen_temp=None):
    """
    Computes temporal basis Psi from the low-dimensional latent Z.
    """
    
    psi_mtx = torch.zeros([K + 1, P], dtype=torch.float).cuda()
    if temporal_basis == 'linear':
        for k in range(K + 1):
            psi_mtx[k, :] = F.interpolate(
                temporal_fcts[:, k, :].view(1, 1, z_dim), size=P,
                mode=temporal_basis)
    elif temporal_basis == 'spline':
        for k in range(K + 1):
            psi_mtx[k, :] = torch_cubic_spline_interp.interp(
                D_t, temporal_fcts[:, k, :].view(z_dim), P_t).view(1, 1, P)
    elif temporal_basis == 'gen':
        psi_mtx = gen_temp(temporal_fcts[0])
    else:
        raise NotImplementedError('Learning mode not implemented.')
    return psi_mtx


class dncnn(torch.nn.Module):
    """
    DnCNN denoiser for RED-PSM framework. 
    """
    
    def __init__(self, numLayers, numChannels, filterSize, est_type='direct'):
        """
        Initializes DnCNN denoiser model.

        Parameters:
        ----------
        numLayers (int): Number of layers.
        numChannels (int): Number of convolutional channels per layer.
        filterSize (int): Filter size per convolutional filter.
        est_type (int): Denoising type. 'direct' or 'residual'.
        """
        super(dncnn, self).__init__()
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=numChannels,
                      kernel_size=filterSize, padding=1), nn.ReLU())
        layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=numChannels, out_channels=numChannels, 
                    kernel_size=filterSize, padding=1),
                nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(
            in_channels=numChannels, out_channels=1, kernel_size=filterSize,
            padding=1)
        self.est_type = est_type
        
    def forward(self, x):
        """
        Performs denoising of the noisy input x.

        Parameters:
        ----------
        x (torch.Tensor): Noisy input frame. 
            Shape: [bs, num_ch, spatial_dim, spatial_dim]

        Returns:
        ----------
        output (torch.Tensor): Denoised image or the estimated noise. 
            Shape: [spatial_dim, spatial_dim]
        """
        if self.est_type == 'direct':
            return self.final_layer(self.main(self.init_layer(x)))
        elif self.est_type == 'residual':
            return x - self.final_layer(self.main(self.init_layer(x)))
    

class dncnnPatchBased_patchLoss(torch.nn.Module):
    """
    Patch-based DnCNN denoiser for RED-PSM with patch-wise denoised output.
    """
    
    def __init__(self, numLayers, numChannels, filterSize, est_type='direct'):
        super(dncnnPatchBased_patchLoss, self).__init__()
        self.filterSize = filterSize
        self.est_type = est_type
        self.init_layer = nn.Sequential(nn.Conv2d(1, numChannels,
                                                  filterSize, 1), nn.ReLU())
        layers = [nn.Sequential(nn.Conv2d(
            numChannels, numChannels, filterSize, 1),
                                nn.ReLU()) for i in range(numLayers)]
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(in_channels=numChannels, out_channels=1,
                                     kernel_size=filterSize, padding=1)
        
    def forward(self, patches):
        if self.est_type == 'direct':
            return self.final_layer(self.main(self.init_layer(patches)))
        elif self.est_type == 'residual':
            return patches - self.final_layer(
                self.main(self.init_layer(patches)))


class RedPsm(nn.Module):
    """
    RED-PSM model class initializing temporal latent representations,
        spatial and temporal basis functions, and the full-rank object f. 
        Supports multiple measurements at a given time instant.
    """
    
    def __init__(self, P, K, spatial_dim, z_dim, temporal_basis, obj_type='walnut',
                 temp_init_type='random', f_init_type='random', 
                 spatial_init_type='random', temporal_mode='z', 
                 noise_std=0, mask=False, rep=4):
        """
        Initializes the RED-PSM model.

        Parameters:
        ----------
        P (int): Number of measurements.
        K (int): PSM order.
        spatial_dim (int): Spatial dimension.
        z_dim (int): Temporal latent representation dimension.
        temporal_basis (str): Basis to map latent representations to temporal 
            basis functions.
        obj_type (str): Object type for initialization.
        temp_init_type (str): Temporal basis initialization type.
        f_init_type (str): Initialization type for object f.
        spatial_init_type (str): Spatial basis initialization type.
        temporal_mode (str): Learn P-dim temporal basis fcts (Psi) or d-dim
            latent z.
        noise_std (float): Measurement noise std for correct initialization.
        mask (bool): If True, apply FOV mask for tomographic objects.
        rep (int): Number of simultaneous measurements/projections.
        """
        super(RedPsm, self).__init__()

        self.P, self.K, self.z_dim = P, K, z_dim
        self.temporal_basis = temporal_basis
        self.f_init_type = f_init_type
        self.spatial_init_type = spatial_init_type
        self.temporal_mode = temporal_mode
        self.rep = rep
                        
        # Cubic spline interpolation
        self.D_t = torch.linspace(0, 1, self.z_dim).cuda()
        self.P_t = torch.linspace(0, 1, self.P//self.rep).cuda()
            
        if temporal_basis in ['linear', 'spline']:
            if temporal_mode == 'z':
                self.psi_mtx = torch.zeros([self.K + 1, self.P//self.rep],
                                           dtype=torch.float).cuda()
                if temp_init_type == 'random':
                    self.temporal_fcts = torch.autograd.Variable(
                        torch.randn(1, K + 1, z_dim).cuda(), requires_grad=True)
                elif temp_init_type == 'learned':
                    self.temp_fcts = torch.randn(1, K + 1, z_dim)
                    self.temp_fcts = torch.load(
                        'data/temporal_latent_fcts_est'
                        '_%s_%s_P_%d_K_%d_L_out_%d_noise_std_%.2e.pt' %(
                        temporal_basis, obj_type, P, K, z_dim, noise_std))
                    self.temporal_fcts = torch.randn(1, K + 1, z_dim).cuda()
                    for k in range(self.K + 1):
                        self.temporal_fcts[:, k, :] = F.interpolate(
                            self.temp_fcts[:, k, :].view(
                                1, 1, self.temp_fcts.shape[2]), 
                            size=self.z_dim, mode='linear')
                    self.temporal_fcts = torch.autograd.Variable(
                        self.temporal_fcts, requires_grad=True)
                else:
                    raise NotImplementedError(
                        'temp_init_type not implemented.')

        # FOV mask
        self.mask = torch.autograd.Variable(
            torch.ones(spatial_dim, spatial_dim).cuda(), requires_grad=False)
        if mask:
            for i,j in list(itertools.product(
                np.arange(spatial_dim), np.arange(spatial_dim))):
                if (i-spatial_dim//2)**2 + (
                    j-spatial_dim//2)**2 >= (spatial_dim//2)**2:
                    self.mask[i,j] = 0
        
        # Spatial basis functions
        if spatial_init_type == 'random':
            self.spatial_basis_fcts = torch.autograd.Variable(
                torch.zeros(
                    K + 1, spatial_dim, spatial_dim).cuda(), requires_grad=True)
        elif spatial_init_type == 'learned':
            self.spatial_basis_fcts = torch.zeros(
                K + 1, spatial_dim, spatial_dim).cuda()
            self.spatial_basis_fcts[:K+1] = torch.load(
                'data/spatial_basis_fcts_est_%s_%s_P_%d_noise_std_%.2e.pt' %(
                'spline', obj_type, P, noise_std)).cuda()[:, :, :K+1].permute(
                2, 0, 1)
            self.spatial_basis_fcts = torch.autograd.Variable(
                self.spatial_basis_fcts, requires_grad=True)
        else:
            raise NotImplementedError('Spatial basis type not implemented.')
        
        f_est = torch.einsum(
            'kp,kjs->pjs', self.psi_mtx, self.mask * self.spatial_basis_fcts)
        self.f_est = torch.autograd.Variable(
            (self.mask * f_est), requires_grad=True)


    def forward(self):
        """
        Computes the PSM estimate of the dynamic object.

        Parameters:
        ----------
        None.

        Returns:
        ----------
        psi_mtx (torch.Tensor): Temporal basis functions. Shape: [K, P]
        f_psm_est (torch.Tensor): PSM estimate of the object f. 
            Shape: [spatial_dim, spatial_dim, P]
        """
        if self.temporal_mode == 'z':
            psi_mtx = generate_psi_from_z(
                self.K, self.P//self.rep, self.temporal_fcts, self.z_dim, 
                self.temporal_basis, self.D_t, self.P_t, gen_temp=None)
        else:
            raise NotImplementedError('Temporal mode not implemented.')

        f_psm_est = torch.einsum(
            'kp,kjs->pjs', psi_mtx, self.mask * self.spatial_basis_fcts)
        f_psm_est = (self.mask * f_psm_est).permute(1,2,0)
        return psi_mtx, f_psm_est
