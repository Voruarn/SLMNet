# SLMNet, ISPRS JPRS 2025</a> </p>

- Paper: [Saliency supervised masked autoencoder pretrained salient location mining network for remote sensing image salient object detection](https://www.sciencedirect.com/science/article/pii/S0924271625001236)


## Abstract

Remote sensing image salient object detection (RSI-SOD), as an emerging topic in computer vision, has significant applications across various sectors, such as urban planning, environmental monitoring and disaster management, etc. In recent years, RSI-SOD has seen significant advancements, largely due to advanced representation learning methods and better architectures, such as convolutional neural networks and vision transformers. While current methods predominantly rely on supervised learning, there is potential for enhancement through self-supervised learning approaches, like masked autoencoder. However, we observed that the conventional use of masked autoencoder for pretraining encoders through masked image reconstruction yields subpar results in the context of RSI-SOD. To this end, we propose a novel approach: saliency supervised masked autoencoder (SSMAE) and a corresponding salient location mining network (SLMNet), which is pretrained by SSMAE for the task of RSI-SOD. SSMAE first uses masked autoencoder to reconstruct the masked image, and then employs SLMNet to predict saliency map from the reconstructed image, where saliency supervision is adopted to enable SLMNet to learn robust saliency prior knowledge. SLMNet has three major components: encoder, salient location mining module (SLMM) and the decoder. Specifically, SLMM employs residual multi-level fusion structure to mine the locations of salient objects from multi-scale features produced by the encoder. Later, the decoder fuses the multi-level features from SLMM and encoder to generate the prediction results. Comprehensive experiments on three public datasets demonstrate that our proposed method surpasses the state-of-the-art methods. Code is available at: https://github.com/Voruarn/SLMNet.


```
## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

@article{FU2025222,
  title = {Saliency supervised masked autoencoder pretrained salient location mining network for remote sensing image salient object detection},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {224},
  pages = {222-234},
  year = {2025},
  issn = {0924-2716},
  author = {Yuxiang Fu and Wei Fang and Victor S. Sheng}
}
```
