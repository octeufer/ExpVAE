# Interventional Explaination Generation

This project explores the idea of CEVAE with application in image domains.

### From Gaussian-VAE to Sigma-VAE

This projects initially explores the KL vanishing problem when training the vanilla VAE. The problems exists when generating images with decoder network, where the generated images are not distinguishable.

![Original](/images/vis_ori_20220130_181151-506456.jpg)
![Reconstruction](/images/vis_recon_20220130_181151-524299.jpg)

Even after attuning the hyper-parameter with [Beta-VAE](https://openreview.net/pdf?id=Sy2fzU9gl), this problem still exists.

This problem has been solved by calibrating the decoder network, which is called [Sigma-VAE](https://arxiv.org/abs/2006.13202)

![Original](/images/vis_ori_20220130_191455-115790.jpg)
![Original](/images/vis_recon_20220130_191455-184894.jpg)