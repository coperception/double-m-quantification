# Double-M-Uncertainty-Quantification
Implementation of paper "Uncertainty Quantification of Collaborative Detection for Self-Driving" [paper](https://arxiv.org/abs/2209.08162), [website](https://coperception.github.io/double-m-quantification/)

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

## Test:

### Test stage:
Train benchmark detectors:
- Lowerbound / Upperbound/ DiscoNet
```bash
    CUDA_VISIBLE_DEVICES=0 make test_no_rsu com=upperbound loss_type=kl_loss_corner_pair_ind logpath=check/check_loss_corner_pair_ind nepoch=25
```

### Compute NLL:
Train benchmark detectors:
- Lowerbound / Upperbound/ DiscoNet
```bash
    CUDA_VISIBLE_DEVICES=0 make prob_measure com=upperbound logpath=check/check_loss_corner_pair_ind
```

## Related works:
- [coperception Github repo](https://github.com/coperception/coperception)

## Related papers:
Double-M Qualification:
```bibtex
@article{Su2022uncertainty,
      author    = {Su, Sanbao and Li, Yiming and He, Sihong and Han, Songyang and Feng, Chen and Ding, Caiwen and Miao, Fei},
      title     = {Uncertainty Quantification of Collaborative Detection for Self-Driving},
      year={2022},
      eprint={2209.08162},
      archivePrefix={arXiv}
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