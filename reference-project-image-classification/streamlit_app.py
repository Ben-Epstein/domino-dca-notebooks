import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd


MODEL = "Domino-ai/vit-base-patch16-224-in21k-food101"
PIPELINE = pipeline("image-classification", model=MODEL)

st.markdown("# Food Classifier")


upload = st.file_uploader("Insert image for classification", type=["png", "jpg"])
c1, c2 = st.columns(2)
if upload is not None:
    im = Image.open(upload)
    c1.header("Input Image")
    c1.image(im)
    c2.header("Prediction")
    preds = pd.DataFrame(PIPELINE(im))[["label", "score"]]
    c2.dataframe(preds)
