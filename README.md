<div align="center">

<h1>CS6323 - DiffMimic: <br> Efficient Motion Mimicking with Differentiable Physics</h1>

---


## About
We implement DiffMimic with [Brax](https://github.com/google/brax): 
><img src="https://github.com/google/brax/raw/main/docs/img/brax_logo.gif" width="158" height="40" alt="BRAX"/>
>
>Brax is a fast and fully differentiable physics engine used for research and development of robotics, human perception, materials science, reinforcement learning, and other simulation-heavy applications.

An environment `mimic_env` is implemented for training and benchmarking. `mimic_env` now includes the following characters:
- [HUMANOID](diffmimic/mimic_envs/system_configs/HUMANOID.py): [AMP](https://github.com/nv-tlabs/ASE/blob/main/ase/data/assets/mjcf/amp_humanoid.xml)-formatted humanoid, used for acrobatics skills.

## Installation
```
conda create -n diffmimic python==3.9 -y
conda activate diffmimic

pip install --upgrade pip
pip install --upgrade "jax[cuda]==0.4.2" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

## Get Started
```shell
python main.py
```

## Visualize
```shell
streamlit run visualize.py
```


## Citation
```
@inproceedings{ren2023diffmimic,
  author    = {Ren, Jiawei and Yu, Cunjun and Chen, Siwei and Ma, Xiao and Pan, Liang and Liu, Ziwei},
  title     = {DiffMimic: Efficient Motion Mimicking with Differentiable Physics},
  journal   = {ICLR},
  year      = {2023},
}
```
## Acknowledgment
- Differentiable physics simulation is done by [Brax](https://github.com/google/brax).
- Early version of the code is heavily based on [Imitation Learning via Differentiable Physics (ILD)](https://github.com/sail-sg/ILD).   
- Motion files are borrowed from [DeepMimic](https://github.com/xbpeng/DeepMimic), [ASE](https://github.com/nv-tlabs/ASE), [AMASS](https://amass.is.tue.mpg.de/), and [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html).
- Characters are borrowed from [DeepMimic](https://github.com/xbpeng/DeepMimic) and [ASE](https://github.com/nv-tlabs/ASE).
- The work is inspired by valuable insights from [SuperTrack](https://montreal.ubisoft.com/en/supertrack-motion-tracking-for-physically-simulated-characters-using-supervised-learning/) and [Spacetime Bound](https://milkpku.github.io/project/spacetime.html).