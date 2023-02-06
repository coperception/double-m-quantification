# Uncertainty Quantification of Collaborative Detection for Self-Driving (ICRA 2023)
[Sanbao Su](https://sanbaosu.netlify.app/), [Yiming Li](https://roboticsyimingli.github.io), [Sihong He](https://scholar.google.com/citations?hl=en&user=jLLDCeoAAAAJ), [Songyang Han](https://songyanghan.com/), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ&hl=en), [Caiwen Ding](https://scholar.google.com/citations?hl=en&user=7hR0r_EAAAAJ), [Fei Miao](http://feimiao.org/index.html)

Implementation of paper "Uncertainty Quantification of Collaborative Detection for Self-Driving" [paper](https://arxiv.org/abs/2209.08162), [website](https://coperception.github.io/double-m-quantification/)

![main](https://github.com/coperception/double-m-quantification/blob/gh-pages/static/images/main.png)

## Abstract:

Sharing information between connected and autonomous vehicles (CAVs) fundamentally improves the performance of collaborative object detection for self-driving. However, CAVs still have uncertainties on object detection due to practical challenges, which will affect the later modules in self-driving such as planning and control. Hence, uncertainty quantification is crucial for safety-critical systems such as CAVs. Our work is the first to estimate the uncertainty of collaborative object detection. We propose a novel uncertainty quantification method, called Double-M Quantification, which tailors a moving block bootstrap (MBB) algorithm with direct modeling of the multivariant Gaussian distribution of each corner of the bounding box. Our method captures both the epistemic uncertainty and aleatoric uncertainty with one inference based on the offline Double-M training process. And it can be used with different collaborative object detectors. Through experiments on the comprehensive CAVs collaborative perception dataset, we show that our Double-M method achieves up to 4.09 times improvement on uncertainty score and up to 3.13% accuracy improvement, compared with the state-of-the-art uncertainty quantification. The results also validate that sharing information between CAVs is beneficial for the system in both improving accuracy and reducing uncertainty.

## Install:
1. Clone this repository.
2. `cd` into the cloned repository.
3. Install `coperception` package with pip:
  ```bash
  pip install -e .
  ```
## Getting started:
Please refer to our docs website for detailed documentations for models: https://coperception.readthedocs.io/en/latest/  
Installation:
- [Installation documentations](https://coperception.readthedocs.io/en/latest/getting_started/installation/)

Download dataset:
- [V2X-Sim](https://coperception.readthedocs.io/en/latest/datasets/v2x_sim/)

## Training

```bash
  cd ./tools/det/
```

### Pretrain stage:

Train benchmark detectors:
- Lowerbound / Upperbound
```bash
    CUDA_VISIBLE_DEVICES=0 make train com=upperbound loss_type=corner_loss logpath=check/check_loss_base nepoch=60
    CUDA_VISIBLE_DEVICES=0 make train com=upperbound loss_type=kl_loss_corner_pair_ind logpath=check/check_loss_base nepoch=80
```

- DiscoNet
```bash
    CUDA_VISIBLE_DEVICES=0 make train_disco_no_rsu loss_type=corner_loss logpath=check/check_loss_base nepoch=60
    CUDA_VISIBLE_DEVICES=0 make train_disco_no_rsu loss_type=kl_loss_corner_pair_ind logpath=check/check_loss_base nepoch=80
```

### Train stage:

Train benchmark detectors:
- Lowerbound / Upperbound
```bash
    CUDA_VISIBLE_DEVICES=0 make mbb_train com=upperbound loss_type=kl_loss_corner_pair_ind logpath=check/check_loss_corner_pair_ind nepoch=25
```

- DiscoNet
```bash
    CUDA_VISIBLE_DEVICES=0 make mbb_train_disco_no_rsu loss_type=kl_loss_corner_pair_ind logpath=check/check_loss_corner_pair_ind nepoch=25
```

Compute the covariance for MBB
```bash
CUDA_VISIBLE_DEVICES=0 make mbb_test_no_rsu com=upperbound loss_type=kl_loss_corner_pair_ind logpath=check/check_loss_corner_pair_ind nepoch=25 
CUDA_VISIBLE_DEVICES=0 make compute_mbb_covar com=upperbound logpath=check/check_loss_corner_pair_ind
```

## Test:

### Test stage:


Train benchmark detectors:
- Lowerbound / Upperbound/ DiscoNet
```bash
    CUDA_VISIBLE_DEVICES=0 make test_no_rsu com=upperbound loss_type=kl_loss_corner_pair_ind logpath=check/check_loss_corner_pair_ind nepoch=25
```

## Related works:
- [coperception Github repo](https://github.com/coperception/coperception)

## Related papers:
Double-M Qualification:
```bibtex
@article{Su2022uncertainty,
      author    = {Su, Sanbao and Li, Yiming and He, Sihong and Han, Songyang and Feng, Chen and Ding, Caiwen and Miao, Fei},
      title     = {Uncertainty Quantification of Collaborative Detection for Self-Driving},
      year={2023},
      booktitle={IEEE International Conference on Robotics and Automation (ICRA)}
  }
}
```

V2X-Sim dataset:
```bibtex
@article{Li_2021_RAL,
  title = {V2X-Sim: A Virtual Collaborative Perception Dataset and Benchmark for Autonomous Driving},
  author = {Li, Yiming and Ma, Dekun and An, Ziyan and Wang, Zixun and Zhong, Yiqi and Chen, Siheng and Feng, Chen},
  booktitle = {IEEE Robotics and Automation Letters},
  year = {2022}
}
```
