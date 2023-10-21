# Named Entity Recognition

Named Entity Recognition (NER), also known as Token Classification, is a popular NLP modeling task in which you fine-tune a model to recognize words, phrases, and concepts from a given body of text. 

[![Alt text](https://github.com/Ben-Epstein/domino-dca-notebooks/blob/main/reference-project-ner/images/ner.png?raw=true)](https://github.com/dominodatalab/reference-project-ner)


This is a very powerful technique that can be applied to most any domain: 
* Extracting executive names from a financial report
* Saving ingredients from a recipe
* Identifying streets, names, and addresses to remove PII from data

In this notebook, we will fine-tune an extremely popular class of models, Bert (in this case, [distil-bert](https://huggingface.co/distilbert-base-cased)), on the task of NER. We will perform NER over the [wikiann](https://huggingface.co/datasets/wikiann/viewer/en) dataset, containing wikipedia articles and entities (known as "spans") pertaining to people (PER), locations (LOC), and organizations (ORG). 


# Set up instructions
This project should run in any standard Domino workspace with GPU acceleration hardware available.


Here is an example setup:

```
FROM quay.io/domino/compute-environment-images:ubuntu20-py3.9-r4.2-domino5.4-gpu

USER ubuntu
COPY requirements.txt .
RUN pip install -r requirements.txt
```

