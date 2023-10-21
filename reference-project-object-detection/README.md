# Fine-Tune a DETR model for ppe object detection.

DETR is a Transformer model architecture designed end-to-end Object Detection tasks. First introduced in [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872): by Nicolas Carion et al[^1], the team succesfully combined a convolutional neural network (CNN) with an encoder-decoder model to fully train a model on object detection. They showed how this architecture could be extended to image segmentation as well, another popular computer vision task.

In this notebook, we will fine-tune this Facebook's DETR model on a ppe (personal protective equiptment) data using Domino. We will also leverage Domino's deep integration with MLFlow to track our model performance and log our resulting model to MLFlow.

The assets available in this project are:

* **finetune.ipynb** - A notebook, illustrating the process of getting DETR from [Huggingface 🤗](https://huggingface.co/facebook/detr-resnet-50) into Domino, and using GPU-accelerated backend for the purposes of fine-tuning it with the cppr object dataset, saving and tracking results in Domino's integrated MLFlow environment.


# Set up instructions
This project should run in any standard Domino workspace with GPU acceleration hardware available.

Here is an example setup:

```
FROM quay.io/domino/compute-environment-images:ubuntu20-py3.9-r4.2-domino5.4-gpu

USER ubuntu
COPY requirements.txt .
RUN pip install -r requirements.txt
```


[^1]: _Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov and Sergey Zagoruyko_

