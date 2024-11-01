# Iris Style Transfer

> **Iris Style Transfer: Enhancing Iris Recognition with Style Features and Privacy Preservation through Neural Style Transfer.**
> Iris texture is widely regarded as a gold standard biometric modality for authentication and identification. The demand for robust iris recognition methods, coupled with growing security and privacy concerns regarding iris attacks, has escalated recently. Inspired by neural style transfer, an advanced technique that leverages neural networks to separate content and style features, we hypothesize that iris texture’s style features provide a reliable foundation for recognition and are more resilient to variations like rotation and perspective shifts than traditional approaches. Our experimental results support this hypothesis, showing a significantly higher classification accuracy compared to conventional features on the OpenEDS dataset. Further, we propose using neural style transfer to mask identifiable iris style features, ensuring the protection of sensitive biometric information while maintaining the utility of eye images for tasks like eye segmentation. This work opens new avenues for iris-oriented, secure, and privacy-aware biometric systems.

This repository is anonymized for review.

## 💁 Usage
1. Contact Meta for the access to the OpenEDS2019 dataset.

2. Create conda environment with `conda env create -f environment.yml` and then activate the environment with `conda activate iris_nst`.

3. For testing the feasibility of iris style feature-based recognition, run `iris_classification.sh`.

    For testing iris style transfer, run `iris_style_transfer.sh`.
    
    For a more interactive play with general neural style transfer and iris style transfer, try `nst.ipynb` and `iris_nst.ipynb`.


## 🔧 Environment
Important libraries and their versions by **November 1st, 2024**:

| Library | Version |
| --- | ----------- |
| Python | 3.12.4 by Anaconda|
| PyTorch | 2.4.1 for CUDA 12.4 |
| Scikit-Learn | 1.4.2 |
| WandB | 0.18.5 |

Others:
- The program should be run a computer with at least 32GB RAM. If run on NVIDIA GPU, a minimum VRAM requirement is 32GB. We obtained our results on a cluster with AMD EPYC 7763 64-Core and 4x NVIDIA A100 80GB PCIe.

- We used [Weights & Bias](https://wandb.ai/site) for figures instead of tensorboard. Please install it (already included in `environment.yml`) and set up it (run `wandb login`) properly beforehand.

## 🗺 Instructions on dataset
It should be noticed that the ground truth labels for the test set of OpenEDS2019 dataset have already been released. Please locate all image folders and label folders according to the arguments of the `read_data` function in `utils.py`.