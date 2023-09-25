import red_psm_utils

def update_metrics(f, f_est, f_psm_est, metrics, best_psnr_f_est, best_f_est):
    ''' Updates accuracy metrics. '''

    metrics['PSNR_f'].append(red_psm_utils.compute_psnr(f, f_est))
    metrics['PSNR_psm'].append(red_psm_utils.compute_psnr(f, f_psm_est))
    metrics['MAE_f'].append(red_psm_utils.compute_mae(f, f_est))
    metrics['MAE_psm'].append(red_psm_utils.compute_mae(f, f_psm_est))
    metrics['SSIM_f'].append(red_psm_utils.compute_ssim(f, f_est))
    metrics['SSIM_psm'].append(red_psm_utils.compute_ssim(f, f_psm_est))
    metrics['HFEN_f'].append(red_psm_utils.compute_hfen(f, f_est))
    metrics['HFEN_psm'].append(red_psm_utils.compute_hfen(f, f_psm_est))

    if best_psnr_f_est < metrics['PSNR_f'][-1]:
        best_psnr_f_est = metrics['PSNR_f'][-1]
        best_f_est = f_est

    return metrics, best_psnr_f_est, best_f_est
