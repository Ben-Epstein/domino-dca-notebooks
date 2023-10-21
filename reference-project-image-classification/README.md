# Fine-Tune a Vision Transformer for food classification

ViT is a Transformer model architecture designed for computer vision tasks. First introduced in [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy et al[^1], the team succesfully leveraged the Transformer architecture, known widely as the de-facto system for NLP tasks, to compete with the performance of state of the art computer vision tasks through CNN (Convolutional Neural Networks). 

In this notebook, we will fine-tune this Google's ViT model on a food classification task using Domino. We will also leverage Domino's deep integration with MLFlow to track our model performance and log our resulting model to MLFlow.

The assets available in this project are:

* **finetune.ipynb** - A notebook, illustrating the process of getting ViT from [Huggingface 🤗](https://huggingface.co/google/vit-base-patch16-224-in21k) into Domino, and using GPU-accelerated backend for the purposes of fine-tuning it with the Food dataset, saving and tracking results in Domino's integrated MLFlow environment.


# Set up instructions
This project should run in any standard Domino workspace with GPU acceleration hardware available.

**Ensure you have enough disk space to load the images (~5GB)**

Here is an example setup:

```
FROM quay.io/domino/compute-environment-images:ubuntu20-py3.9-r4.2-domino5.4-gpu

USER ubuntu
COPY requirements.txt .
RUN pip install -r requirements.txt
```

[^1]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
