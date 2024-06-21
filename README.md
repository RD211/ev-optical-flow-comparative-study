# Comparative Study of Model-based and Learning-based Optical Flow Estimation Methods with Event Cameras

## David Dinucu-Jianu
### Thesis committee: Nergis Tömen, Hesam Araghi, Guohao Lan

## Abstract
Optical flow estimation with event cameras encompasses two primary algorithm classes: model-based and learning-based methods. Model-based approaches do not require any training data, while learning-based approaches utilize datasets of events to train neural networks. This study compares model-based and learning-based optical flow estimation methods using event cameras, aiming to provide guidance for real-world applications. We evaluated these methods on the MVSEC and DSEC datasets, focusing on their accuracy, runtime, and limitations.

## Models
In this repository, inside the `models/` folder, there exist three retrained models on the BlinkFlow dataset. The models are TMA and IDNet variants.

## Included Submodules
We have included all methods compared and benchmarked in this study as submodules for easy access and reproducibility:

- [MultiCM](https://github.com/username/multicm): Model-based algorithm that excels in scenarios with small motions.
- [Brebion et al.](https://github.com/username/brebion): Real-time optical flow method optimized for both low- and high-resolution event cameras.
- [E-RAFT](https://github.com/username/eraft): Learning-based method utilizing voxel grid representation for event data.
- [TMA](https://github.com/username/tma): Temporal Motion Aggregation method for event-based optical flow.
- [IDNet](https://github.com/username/idnet): Iterative Deblurring Network optimized for efficiency and accuracy.
- [TamCM](https://github.com/username/tamcm): Taming Contrast Maximization for self-supervised learning of optical flow.
