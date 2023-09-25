import torch


def f_update(
    lmbda, beta, denoiser, patchifier, f, f_psm, gamma, denoiser_type, P):
    """
    Performs the ADMM 2nd primal variable update for `f`.

    The update of `f` is based on the specified `denoiser_type`: patch-based or 
    full-image based. 

    Parameters
    ----------
    lmbda (float): RED regularization weight.
    beta (float): Augmented Lagrangian weight.
    denoiser (function): Denoiser function to be applied.
    patchifier (function): Function to patchify and combine patches of an image.
    f (torch.Tensor): Input data to be denoised.
    f_psm (torch.Tensor): PSM of `f`.
    gamma (torch.Tensor): Dual variable for `f`.
    denoiser_type (str): Type of denoiser, either 'patch_based' or 'full_img'.
    P (int): Total number of instances.

    Returns
    -------
    f (torch.Tensor): The updated value of `f`.
    """
    if denoiser_type == 'patch_based':
        for t in range(P):
            f[t] = denoiser(f[t][None, None, :, :]).squeeze()
        return (lmbda * f + beta * (f_psm + gamma)) / (lmbda + beta)
    elif denoiser_type == 'patch_based_patchloss':
        for t in range(P):
            f[t] = patchifier.merge(
                denoiser(patchifier(f[t][None, None, :, :]))).squeeze()
        return (lmbda * f + beta * (f_psm + gamma)) / (lmbda + beta)
    elif denoiser_type == 'full_img':
        return (
            lmbda * denoiser(f[:, None, :, :]).squeeze() + beta * (
                f_psm + gamma)) / (lmbda + beta)
    else:
        raise NotImplementedError(
            'ADMM f update not implemented for the denoiser type.')


def dual_variable_update(gamma, f, f_psm):
    ''' Performs ADMM dual variable update for gamma. 
    
    Parameters
    ----------
    f (torch.Tensor): Full-rank representation of the object.
    f_psm (torch.Tensor): PSM representation of f.
    gamma (torch.Tensor): Dual variable associated with the constraint f=f_psm.

    Returns
    -------
    gamma (torch.Tensor): The updated value of the dual variable.
    
    '''
    return gamma + f_psm - f


def learn_psm_bases(model, optimizer, scheduler, criterion, theta_exp, g,
                    num_primal_iter, P, spatial_dim, R=None, temporal_mode='z',
                    gamma_est=None, beta=1, xi=1, chi=1, rep=4):
    '''
    Perform RED-PSM primal updates for the PSM basis functions simultaneously
    with multiple simultaneous projections support.

    Parameters:
    ----------
    model (nn.Module): The forward model function that generates temporal 
        functions and estimated object f.
    optimizer: The optimizer used for gradient descent.
    scheduler: Learning rate scheduler.
    criterion: The loss function for the fidelity term between estimated and 
        ground truth measurements.
    theta_exp (list or array): The set of acquisition view angles.
    g (torch.Tensor): The ground truth measurements.
    num_primal_iter (int): Number of primal updates for each basis function.
    P (int): Number of measurements.
    spatial_dim (int): Spatial dimension.
    R (torch.Tensor): Forward operator for measurement acquisition.
    temporal_mode (str): Temporal mode ('z' or 'psi').
    gamma_est (torch.Tensor): The dual variable.
    beta (float): Parameter for the variable split loss.
    xi (float): Parameter for the temporal Frobenius norm penalty.
    chi (float): Parameter for the spatial Frobenius norm penalty.
    rep (int): Number of simultaneous projections.

    Returns:
    ----------
    loss_epoch (dict): A dictionary containing loss values for each epoch.
    f_psm_est (torch.Tensor): Estimated object PSM after training.
    g_f_est (torch.Tensor): Estimated measurements from the estimated object.
    spatial_basis_fcts (torch.Tensor): Spatial basis functions.
    temporal_fcts (torch.Tensor): Temporal latent representations.
    psi_mtx (torch.Tensor): Temporal basis functions.
    '''
    
    # Initialize loss vectors
    loss_epoch = {}
    [loss_epoch['total'], loss_epoch['temp_reg'], loss_epoch['spatial_reg'],
     loss_epoch['var_split'], loss_epoch['wt_reg'], loss_epoch['SSIM_f'],
     loss_epoch['MAE_f'], loss_epoch['f'], loss_epoch['g']] = [[] for _ in range(9)]
    loss_epoch['PSNR_f'] = [-1e4]
    
    # Convert view angle and measurement inputs to torch
    g_gt = torch.Tensor(g).cuda()
        
    # Training
    for _ in range(num_primal_iter):
        # Call forward model to generate temporal fcts and estimated object f
        psi_mtx, f_psm_est = model()
        f_psm_est = f_psm_est.permute(2, 0, 1).clip(min=0)

        # Frobenius norm penalty for temporal fcts.
        if temporal_mode == 'z':
            T = psi_mtx
        elif temporal_mode == 'psi':
            T = model.psi_mtx
        else:
            raise NotImplementedError('temporal_mode not implemented.')
        loss_temporal = xi * T.pow(2).mean()

        # Frobenius norm penalty for spatial fcts.
        loss_spatial = chi * model.spatial_basis_fcts.pow(2).mean()
        
        # Compute projections from the estimated object
        g_f_est = torch.einsum('pjs,ps->jp', R, torch.repeat_interleave(
            f_psm_est, repeats=rep, dim=0).view(P, spatial_dim**2))
        
        # Compute the data fidelity
        loss_g = criterion(g_f_est, g_gt)
        
        # Compute the Lagrangian
        loss_var_split = (beta / 2) * torch.linalg.norm(
            f_psm_est - model.f_est + gamma_est)
        
        loss = loss_g + loss_var_split + loss_temporal + loss_spatial
        
        # Backprop
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        scheduler.step()
        
        # Log computed loss values
        loss_epoch['total'].append(loss.data.cpu().numpy())
        loss_epoch['temp_reg'].append((loss_temporal).data.cpu().numpy())
        loss_epoch['spatial_reg'].append((loss_spatial).data.cpu().numpy())
        loss_epoch['g'].append(loss_g.data.cpu().numpy())
        loss_epoch['var_split'].append(loss_var_split.data.cpu().numpy())

    return [loss_epoch, f_psm_est, g_f_est, model.spatial_basis_fcts,
            model.temporal_fcts, psi_mtx]
