# OUI Need to Talk About Weight Decay

📄 [Read the paper on arXiv](https://arxiv.org/abs/2504.17160)  
🧠 [Try the code on GitHub](https://github.com/AlbertoFdezHdez/OUI)

---

## 🧪 Overview

We introduce **OUI** (Overfitting–Underfitting Indicator), a simple yet powerful diagnostic tool to monitor the training dynamics of deep neural networks and identify the optimal value for **weight decay** — without using a validation set.

Unlike traditional validation-based hyperparameter tuning, OUI:
- Converges faster than metrics like accuracy or loss.
- Signals overfitting and underfitting through the internal statistics of the model.
- Provides a reliable early criterion for choosing the regularization strength during training.

---

## 📊 Results

OUI was validated across multiple vision benchmarks:

- **DenseNet-BC-100** on **CIFAR-100**
- **EfficientNet-B0** on **TinyImageNet**
- **ResNet-34** on **ImageNet-1K**

Across all experiments, keeping OUI between 0.6 and 0.8 consistently led to improved validation accuracy.

