# PFL-DKD

This repository contains the official implementation for the manuscript:
> [Modeling Knowledge Fusion with Distillation for Improving Personalized Federated Learning](null)

Personalized federated learning (PFL) develops a customized model for each local client to mitigate accuracy issues caused by data and system heterogeneity of different local clients. Most of the works on PFL adopt a centralized federation and, therefore, suffer from single-point failures or malicious attacks on the global server.  In addition,  due to system heterogeneity, such as different computing capabilities of local clients, the computational and storage constraints vary significantly among local clients, which limits the centralized model aggregation considerably. To tackle the issues mentioned above, we develop a novel PFL framework, called PFL-DKD, by modeling an intrinsic \textbf{decoupling of knowledge distillation}(DKD) mechanism to spin out personalization. In PFL-DKD, multiple local clients are in a dynamically connected topology, and the local clients are divided into clients with different knowledge based on computational and storage capabilities. Local clients transfer learning from knowledge-rich clients to knowledge-poor clients using decoupled knowledge distillation in a peer-to-peer convergent manner. We extend PFL-DKD to PFL-FDKD, by plugging \textbf{multilogit fusion}, where the knowledge and experiences of all neighbors are seamlessly aggregated into the teacher logit. Comprehensive experiments demonstrate that our proposed methods outperform centralized and decentralized PFL baselines while significantly mitigating the challenges of heterogeneous data and the system. The details of our implementation with the codebase are in [PFL-DKD](https://github.com/Luck12138/PFL-DKD.git).

# Experiments
The implementations of each method are provided in the folder `/fedml_api/standalone`, while experiments are provided in the folder `/fedml_experiments/standalone`.

Use dataset corresponding bash file to run the experiments.

```
cd /fedml_experiments/standalone/pfldkd
```

<!-- # Citation

If you find this repo useful for your research, please consider citing the paper

```
@InProceedings{pmlr-v162-dai22b,
  title = 	 {{D}is{PFL}: Towards Communication-Efficient Personalized Federated Learning via Decentralized Sparse Training},
  author =       {Dai, Rong and Shen, Li and He, Fengxiang and Tian, Xinmei and Tao, Dacheng},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {4587--4604},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/dai22b/dai22b.pdf},
  url = 	 {https://proceedings.mlr.press/v162/dai22b.html},
  abstract = 	 {Personalized federated learning is proposed to handle the data heterogeneity problem amongst clients by learning dedicated tailored local models for each user. However, existing works are often built in a centralized way, leading to high communication pressure and high vulnerability when a failure or an attack on the central server occurs. In this work, we propose a novel personalized federated learning framework in a decentralized (peer-to-peer) communication protocol named DisPFL, which employs personalized sparse masks to customize sparse local models on the edge. To further save the communication and computation cost, we propose a decentralized sparse training technique, which means that each local model in DisPFL only maintains a fixed number of active parameters throughout the whole local training and peer-to-peer communication process. Comprehensive experiments demonstrate that DisPFL significantly saves the communication bottleneck for the busiest node among all clients and, at the same time, achieves higher model accuracy with less computation cost and communication rounds. Furthermore, we demonstrate that our method can easily adapt to heterogeneous local clients with varying computation complexities and achieves better personalized performances.}
}
``` -->

<!-- [//]: # (## Citation)

[//]: # ()
[//]: # (If you find this repo useful for your research, please consider citing the paper)

[//]: # (```)

[//]: # (@article{yang2021class,)

[//]: # (  title={Class-Disentanglement and Applications in Adversarial Detection and Defense},)

[//]: # (  author={Yang, Kaiwen and Zhou, Tianyi and Tian, Xinmei and Tao, Dacheng and others},)

[//]: # (  journal={Advances in Neural Information Processing Systems},)

[//]: # (  volume={34},)

[//]: # (  year={2021})

[//]: # (})

[//]: # (```) -->
