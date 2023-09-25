from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def plot_and_save_red_psm_results(params, obj_type, results_per_config, P, psm_update_type, noise_std, ang_period, save_fig):
    STR_PERIOD = '' if ang_period is None else '_' + str(ang_period)
    PATH = 'data/red_psm_results/%s_%d%s_denoiser_%s_%s/' %(
        obj_type, P, STR_PERIOD, params['denoiser_type'], 
        psm_update_type)
    if params['denoiser_type'] == 'patch_based':
        PATH = PATH + '_patch_sz_' + str(
            params['pSize_sweep']) + '_patch_str_' + str(
            params['pStride_sweep']) + '_num_layers_' + str(
            params['num_layers_sweep']) + '_num_ch_' + str(
            params['num_channels'])
    PATH += 'data/'
    Path(PATH).mkdir(parents=True, exist_ok=True)

    print('Results with experimental order:\n')
    for res in results_per_config:
        print(res[0], 'PSNR:', res[1]['PSNR_f'][-1],
              'SSIM:', res[1]['SSIM_f'][-1],
              'MAE:', res[1]['MAE_f'][-1],
              'HFEN:', res[1]['HFEN_f'][-1], res[3])

    results_per_config_sorted = sorted(results_per_config, key=lambda res: res[0])

    cnt = 0
    plt.figure(figsize=(20,3))
    for res in results_per_config_sorted:
        label_str = (
            r'$K$:%d $d$:%d $\beta$:%.1e $\lambda$:%.1e $\xi$:%.1e $\chi$:%.1e' %(
                res[3]['K'], res[3]['z_dim'], res[3]['beta'], res[3]['lmbda'], 
                res[3]['xi'], res[3]['chi']) + '\npsz/pstr/layers:%d/%d/%d' %(
                res[3]['pSize'], res[3]['pStride'], res[3]['num_layers']))
        if cnt % 3 == 0:
            LINESTYLE='--'
        elif cnt % 3 == 1:
            LINESTYLE='-'
        else:
            LINESTYLE='-.'
        iter = params['acc_compute_freq'] * np.arange(len(res[1]['PSNR_f']))
        plt.subplot(1,4,1); plt.title('PSNR (dB)')
        plt.plot(iter, res[1]['PSNR_f'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.xlabel('Iter')
        plt.subplot(1,4,2); plt.title('SSIM')
        plt.plot(iter, res[1]['SSIM_f'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.xlabel('Iter')
        plt.subplot(1,4,3); plt.title('MAE')
        plt.plot(iter, res[1]['MAE_f'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.xlabel('Iter')
        plt.subplot(1,4,4); plt.title('HFEN')
        plt.plot(iter, res[1]['HFEN_f'], label=label_str, linestyle=LINESTYLE)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.xlabel('Iter')
        plt.legend(bbox_to_anchor=(0.75, -0.25), ncol=3)
        cnt += 1
    if save_fig:
        save_fig_path = (
            'PSNR_SSIM_MAE_HFEN_P_%d_%s_num_epoch_%d'
            '_num_primal_iter_%d_noise_std_%.2e'
            '_lr_%.2e_num_exps_%d_init_%s' %(
                P, res[3]['temporal_basis'], params['num_epoch'], 
                params['num_primal_iter'], noise_std, params['lr_primal'], 
                len(results_per_config_sorted), params['temp_fcts_type']))
        plt.savefig(PATH + save_fig_path + '.jpg', bbox_inches='tight')
        plt.savefig(PATH + save_fig_path + '.pdf', bbox_inches='tight')
    plt.show()

    print('Results with increasing PSNR:\n')
    for res in results_per_config_sorted:
        print(
            'PSNR:', res[0],
            'SSIM:', res[1]['SSIM_f'][np.argmax(np.array(res[1]['PSNR_f']))], 
            'MAE:', res[1]['MAE_f'][np.argmax(np.array(res[1]['PSNR_f']))],
            'HFEN', res[1]['HFEN_f'][np.argmax(np.array(res[1]['PSNR_f']))]
        )
        if save_fig:
            save_data_path = (
                '_K_%d_d_%d_beta_%.2e_lambda_%.2e_xi_%.2e_chi_%.2e'
                '_PSNR_%.2e_pSize_%d_pStride_%d_numlayers_%d_init_%s.npy' %(
                    res[3]['K'], res[3]['z_dim'], res[3]['beta'],
                    res[3]['lmbda'], res[3]['xi'], res[3]['chi'], res[0], 
                    res[3]['pSize'], res[3]['pStride'], res[3]['num_layers'], 
                    params['temp_fcts_type']))
            np.save(PATH + 'f_est' + save_data_path, res[4]['f_est'])
            np.save(PATH + 'spatial_basis_est' + save_data_path,
                    res[4]['spatial_basis_fcts'])
            np.save(PATH + 'temporal_basis_est' + save_data_path,
                    res[4]['temporal_latent_fcts'])
            np.save(PATH + 'psi_est' + save_data_path,
                    res[4]['psi_mtx'].detach().cpu().numpy())
    pass


def train_visualization(loss_epoch, f_est, f_psm_est, gamma_bar_est, f, t):
    plt.figure(figsize=(18,2))
    plt.subplot(1,8,1)
    plt.title(r'$\log_{10}(H(f, \Lambda, \Psi))$')
    plt.plot(np.log10(loss_epoch['total']))
    plt.grid()
    plt.subplot(1,8,2)
    plt.title(r'$\log_{10}(\|\Psi\|_{F}^2)$')
    plt.plot(np.log(loss_epoch['temp_reg']))
    plt.grid()
    plt.subplot(1,8,3)
    plt.title(r'$\log_{10}(\sum_{t}\|g_{t,est} - g_t\|_2^2/P)$')
    plt.plot(np.log(loss_epoch['g']))
    plt.grid()
    plt.subplot(1,8,4)
    plt.title(r'$f_{est}$')
    plt.imshow(f_est[t])
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,8,5)
    plt.title(r'$f_{PSM}$')
    plt.imshow(f_psm_est[t])
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,8,6)
    plt.title(r'$\gamma_{est}$')
    plt.imshow(gamma_bar_est[t])
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,8,7)
    plt.title(r'$f$')
    plt.imshow(f[..., t])
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,8,8)
    plt.title(r'$f-f_{est}$')
    plt.imshow(f_est[t].numpy()-f[..., t])
    plt.axis('off')
    plt.colorbar()
    plt.show()
    pass


def plot_psm_basis_fcts(psi_mtx, spatial_basis_fcts, K):
    fig, ax = plt.subplots(1, K+1, figsize=(3*(K+1), 3))
    ax[0].set_ylabel(r'$\Psi$')
    for k in range(K+1):
        ax[k].plot(psi_mtx[k].detach().cpu())
        ax[k].tick_params(left=False, right=False , labelleft=False,
                labelbottom=False, bottom=False)
    plt.show()
    
    fig, ax = plt.subplots(1, K+1, figsize=(3*(K+1), 3))
    ax[0].set_ylabel(r'$\Lambda$')
    for k in range(K+1):
        ax[k].imshow(spatial_basis_fcts[k].detach().cpu())
        ax[k].tick_params(left=False, right=False , labelleft=False,
                labelbottom=False, bottom=False)
    plt.show()
    pass


def display_f(f, image, rate, P):
    Psqrt = np.int(np.ceil(np.sqrt(P / rate)))
    plt.figure(figsize=(5, 5))
    for p in range(P):
        if p % rate == 0:
            plt.subplot(Psqrt, Psqrt, p // rate + 1)
            plt.imshow(f[..., p], cmap='gray')
            plt.clim(0, np.max(image))
    plt.show()
    

def display_inputs(f, theta_exp, P, num_frames):
    num_frames = 8
    fig, ax = plt.subplots(1, num_frames, figsize=(2.5*num_frames, 3))
    plt.suptitle(r'Ground-truth frames $f_t$ for $t \in [0, 1]$')
    for k in range(num_frames):
        ax[k].imshow(f[..., k*P//num_frames], cmap='gray', clim=(0,f.max()))
        ax[k].set_title(r'$t = %d/%d$' %(k*P//num_frames, P))
        ax[k].axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(2.5*num_frames, 3))
    plt.suptitle(r'Acquisition scheme $\{\theta(t)\}$ for $t \in [0, 1]$')
    plt.scatter(np.arange(len(theta_exp))/len(theta_exp), theta_exp, s=10)
    plt.ylabel(r'$\theta(t)$, $rad$')
    plt.xlabel(r'$t$, $sec$')
    plt.grid(linestyle='--', linewidth=0.25, which='major', alpha=0.75)
    plt.show()