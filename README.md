# Vector Field Transformer
### [Medical Image Analysis 2023] Domain Generalization for Retinal Vessel Segmentation via Hessian-based Vector Field 
---
### Introduction
Blessed by the vast amount of data, the learning-based methods have achieved remarkable performance in almost all tasks in computer vision and medical image analysis. Although these deep models can simulate highly nonlinear mapping functions, they are not robust with regard to domain shift of input data. This is one of the major concerns that impedes the large scale deployment of deep models in medical images since they have inherent variation in data distribution due to lack of imaging standardization. Therefore, a domain generalization (DG) method is needed to alleviate this problem. In this work, our main contributions are in three folds:
>- **data augmentation:** A full-resolution variational auto-encoder (f-VAE) network that generates synthetic latent images
>- **domain alignment:** A Hessian-based vector field that serves as an aligned image space that delineates the morphology of vessels
>- **model architecture:** A novel paralleled transformer blocks that helps to learn local features in different scales

The overall pipeline of the work is shown as following:
<p align="center">
  <img src="/assets/pipeline.png" alt="drawing" width="500"/>
</p>

### Data augmentation
We implement a full resolution variational auto-encoder (f-VAE) in which the latent space is set to have the same width and height with the input image. The architecture is illustrated by the image below:
<p align="center">
  <img src="/assets/augment_network.png" alt="drawing" width="250"/>
</p>

The supervision is provided by the binary vessel map. Since there is no direct constraint on the latent representation, we observe that each time we re-train the f-VAE will result in latent images with different styles while the anatomical structure remains unchanged. The <ins>augment-2</ins>, <ins>augment-3</ins> and <ins>augment-4</ins> are the three synthesized versions while augment-1 is the CLAHE version (intensty inversed) of the raw input.
<p align="center">
  <img src="/assets/aug_results.png" alt="drawing" width="700"/>
</p>

The code for the synthetic model is accessible [here](https://github.com/MedICL-VU/Vector-Field-Transformer/tree/main/src/augmentation)  
