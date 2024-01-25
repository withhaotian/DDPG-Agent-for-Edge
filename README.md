# CoPace: Edge Computation Offloading and Caching for Self-Driving With Deep Reinforcement Learning
This repository is the implementation of the deep reinforcement learning (DRL) agent of Deep Deterministic Policy Gradient (DDPG) proposed in the paper "CoPace: Edge Computation Offloading and Caching for Self-Driving With Deep Reinforcement Learning," in *IEEE Transactions on Vehicular Technology* (2021). This myDDPG.py is a tailored DRL agent and my_env.py is the DRL environment for the CoPace algorithm. The CoPace algorithm is designed to realize the joint computation offloading, content caching, and resources allocation (including computation and communication) for self-driving vehicles in edge computing systems.

## Quick start
You can integrate myDDPG agent of CoPace into your own DRL env to sovle the optimization problems in edge computing with your formulated system models.

## Datasets Fetching
In the CoPace algorithm, we use two datasets, i.e., telecom datasets [1] in Shanghai includes more than 7.2 million accessing histories through more than 3200 base stations from Jun 1, 2014 to Nov 30, 2014, and taxi trajectory datasets [2] with GPS reports on Feb 2007 in Shanghai, China. If you are intersted in these datasets, you can access them from the following papers:

[1] Wang S, Guo Y, Zhang N, et al. Delay-aware microservice coordination in mobile edge computing: A reinforcement learning approach[J]. IEEE Transactions on Mobile Computing, 2019, 20(3): 939-951.

[2] Liu S, Liu Y, Ni L M, et al. Towards mobility-based clustering[C]//Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining. 2010: 919-928.

## Requirements
Tensorflow == 1.15.0

Python == 3.7

## Cite this work
```bibtex
@ARTICLE{9580706,
  author={Tian, Hao and Xu, Xiaolong and Qi, Lianyong and Zhang, Xuyun and Dou, Wanchun and Yu, Shui and Ni, Qiang},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={CoPace: Edge Computation Offloading and Caching for Self-Driving With Deep Reinforcement Learning}, 
  year={2021},
  volume={70},
  number={12},
  pages={13281-13293},
  doi={10.1109/TVT.2021.3121096}}
```

## Contact us
[Hao Tian](https://withhaotian.github.io), Nanjing University, withhaotian [at] gmail.com