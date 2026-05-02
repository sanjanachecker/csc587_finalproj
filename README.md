# CSC 587 Final Project
Sanjana Checker and Sofija Dimitrijevic

## Introduction
Our project will include 2 challenges, the first of which is land cover classification: identifying what class each pixel is in (e.g. forest, urban, water, etc.) This is a very important problem in remote sensing and has applications in disaster relief, urban planning, and environmental monitoring. Semantic segmentation of satellite imagery is non trivial because models need to understand both global context and fine grained spatial detail at once. The next challenge is deployment: even if a segmentation model is accurate, running it efficiently on edge hardware like drones and satellites requires a small and fast model. We will explore post training quantization (PTQ) to reduce model weights from 32-bit floats to 8-bit integers. We will then measure the accuracy-latency-size tradeoff. 
The scope of our project includes training a CNN on a satellite dataset, and then exploring post-training quantization as a compression technique and measuring the tradeoffs. The dual focus in this project makes it both practically relevant and experimentally informative. 

## Related Work
First, the U-Net paper (Ronneberger et al., 2015) introduced the encoder-decoder architecture with skip connections. This became the main approach for biomedical and satellite image segmentation, and is still a strong baseline today. EfficientNet (Tan & Le, 2019) proposed a compound scaling of CNN width, depth, and resolution. It is used as a lightweight U-Net encoder backbone. 
As far as segmentation goes, EuroSAT (Helber et al., 2019) introduced benchmark datasets for satellite land cover classification. EuroSAT showed that CNNs trained on Sentinel-2 imagery achieve very strong land use classification performance. Zhao et al. (2023) provide a survey of deep learning approaches to land use and land cover (LULC) classification. They cover CNN, autoencoder, GAN, and RNN approaches across both pixel-level and patch-level classification tasks. The survey shows that even though deep learning has improved LULC performance, there are many challenges around high dimensional remote sensing data and limited labeled samples. 
Finally, the foundational paper on INT8 quantization of deep networks (Jacob et al., 2018) showed that 8-bit inference can match 32-bit accuracy with minimal degradation on vision tasks. 

## Method
For our dataset, we aim to use EuroSAT, which is available as a direct download, no GEE needed. EuroSAT has 27,000 labeled 64x64 sentinel-2 image patches across 10 land cover classes. We will use the RGB subset for simplicity, with an 80/10/10 train/val/test split. 
We will implement a U-Net in PyTorch from scratch with a lightweight EfficientNet-B0 encoder pretrained on imagenet, and a custom decoder with skip connections. This will be part of our novel code contribution since we’re not using a segmentation library end-to-end, we’re building the decoder and training loop ourselves. 
For training, we will use cross-entropy loss with class weighting to handle any imbalance - this part might take a lot of trial/error. We’ll use Adam optimizer and 30-50 epochs on Colab (should be fine since EuroSAT has small image sizes). 
After training is complete, we will apply PyTorch’s built in PTQ (torch.quantization) to convert the model down to INT8. We will also try out dynamic quantization as a comparison point. We won’t do any quantization-aware training so that we can really isolate the compression effect. 
For experiments, we will run the following ablations:
FP32 baseline vs INT8 static vs INT8 dynamic
Accuracy (per class and overall), inference latency (ms/image), and model size (MB) for each
Optional: switch out EfficientNet-B0 for MobileNetV2 encoder and repeat

## Evaluation
Since our project has two phases, we will evaluate using a few strategies for each focus. First, we will evaluate the semantic segmentation quality using well-known metrics, such as mIoU (mean intersection over union), per-class IoU, and more simple pixel accuracy (depending on how much class imbalance we have). mIoU is our primary metric that calculates the overlap between the ground truth and predicted masks for each class (pixel type), averaging for all classes. Zhao et al. discuss this as one of their primary metrics, along with overall accuracy, average accuracy, and F1-score, which we will also use as our evaluation metrics. Finally, we will include a full class confusion matrix to visualize misclassifications.

For the PTQ focus, we will measure the tradeoff curve based on three configurations: FP32 baseline, INT8 static quantization, and INT8 dynamic quantization. Our quantization approach will follow the integer arithmetic inference framework introduced in Jacob et al. For every configuration, we will report on accuracy through mIoU and ΔmIoU (measures explicit drop relative to FP32 baseline). We will also report which classes suffer the most from quantization by including per-class IoU degradation. For efficiency measurement, we will report on model size (in MB), which is the static model size before and after quantization. Jacob et al. found a roughly 4x compression from FP32 to INT8, and we will verify this holds for the EfficientNet-B0 encoder backbone (Tan & Le, 2019). We will also measure latency (ms/image) on the CPU and peak runtime memory footprint (RAM usage during inference). 

We will have side by side visualizations of segmentation overlays consisting of FP32 vs INT8 predictions on the same image to make differences visually interpretable in the final report. 

## References:

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 9351, 234–241. https://arxiv.org/abs/1505.04597

Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217–2226. https://arxiv.org/abs/1709.00029

Zhao, S., Tu, K., Ye, S., Tang, H., Hu, Y., & Xie, C. (2023). Land Use and Land Cover Classification Meets Deep Learning: A Review. Sensors, 23(21), 8966. https://doi.org/10.3390/s23218966

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. Proceedings of the 36th International Conference on Machine Learning (ICML), 97, 6105–6114. https://arxiv.org/abs/1905.11946

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2704–2713. https://arxiv.org/abs/1712.05877

