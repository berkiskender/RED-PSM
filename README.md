# RED-PSM

Implementation of *RED-PSM: Regularization by Denoising of Partially Separable Models for Dynamic Imaging* ([IEEE TCI 2024](https://ieeexplore.ieee.org/document/10535218), [arXiv](https://arxiv.org/abs/2304.03483), [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Iskender_RED-PSM_Regularization_by_Denoising_of_Partially_Separable_Models_for_Dynamic_ICCV_2023_paper.pdf), [5 min video](https://youtu.be/jdWdY9XJ0Ew))

*Berk Iskender, Marc L. Klasky, Yoram Bresler*

Dynamic imaging addresses the recovery of a time-varying 2D or 3D object at each time instant using its undersampled measurements. In particular, in the case of dynamic tomography, only a single projection at a single view angle may be available at a time, making the problem severely ill-posed. In this work, we propose an approach, RED-PSM, which combines for the first time two powerful techniques to address this challenging imaging problem. The first, are partially separable models, which have been used to efficiently introduce a low-rank prior for the spatio-temporal object. The second is the recent Regularization by Denoising (RED), which provides a flexible framework to exploit the impressive performance of state-of-the-art image denoising algorithms, for various inverse problems. We propose a partially separable objective with RED and an optimization scheme with variable splitting and ADMM, and prove the convergence of our objective to a value corresponding to a stationary point satisfying the first-order optimality conditions. Convergence is accelerated by a particular projection-domain-based initialization. We demonstrate the performance and computational improvements of our proposed RED-PSM with a learned image denoiser by comparing it to a recent deep-prior-based method TD-DIP.

![alt text](https://github.com/berkiskender/RED-PSM/blob/master/red_psm.jpeg?raw=true)

## Parameter configurations
RED-PSM training parameters are stored and can be modified at ```red_psm_train_cfg.yaml```.
Other hyperparameters & flags can be set in the main notebook script ```red_psm.ipynb```.

The default configuration is for the dynamic walnut object with total number of views P=256. Configurations for other settings are reported in the supplementary material of the manuscript.

## RED denoiser
Two pre-trained DnCNN denoisers for the dynamic walnut object is provided in ```data/denoiser```. 
The denoiser code can be found in ```red_psm_models.py```. 
If required, denoisers pre-trained on different objects/distributions can also be incorporated using the denoiser definition in ```red_psm_models.py```.

## Forward Model
To download tomographic forward models (measurement operators) for different total number of measurements (32, 64, 128, and 256), run
```shell
cd forward_model
bash forward_model_download_P32.sh
bash forward_model_download_P64.sh
bash forward_model_download_P128.sh
bash forward_model_download_P256.sh
```
Different forward models for different imaging modalities with dimensions ```[measurement size x image size x total number of measurements]``` can also be used with RED-PSM.

## Citation
If you find RED-PSM useful for your research, please cite:

*B. Iskender, M. L. Klasky and Y. Bresler, "RED-PSM: Regularization by Denoising of Partially Separable Models for Dynamic Imaging," 2023 IEEE/CVF International Conference on Computer Vision (ICCV), Paris, France, 2023, pp. 10561-10570, doi: 10.1109/ICCV51070.2023.00972.*

```
@INPROCEEDINGS{iskender2023red,
  author={Iskender, Berk and Klasky, Marc L. and Bresler, Yoram},
  booktitle={2023 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  title={RED-PSM: Regularization by Denoising of Partially Separable Models for Dynamic Imaging}, 
  year={2023},
  volume={},
  number={},
  pages={10561-10570},
  doi={10.1109/ICCV51070.2023.00972}}
```

## Contact
In case of any questions, feel free to contact via email: Berk Iskender, berki2@illinois.edu.
