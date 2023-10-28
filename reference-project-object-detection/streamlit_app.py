import pandas as pd
import streamlit as st
from PIL import Image
from transformers import pipeline

from score import draw_boxes

MODEL = "devonho/detr-resnet-50_finetuned_cppe5"
PIPELINE = pipeline("object-detection", model=MODEL)

st.markdown("# Food Classifier")


upload = st.file_uploader("Insert PPE image for object detection", type=["png", "jpg"])
c1, c2 = st.columns(2)
if upload is not None:
    im = Image.open(upload)
    c1.header("Input Image")
    c1.image(im)
    c2.header("Prediction")
    objects = PIPELINE(im)
    objects_df = pd.DataFrame(objects)
    objects_df[["xmin", "ymin", "xmax", "ymax"]] = objects_df.pop("box").apply(
        pd.Series
    )
    obejcts_image = draw_boxes(im, objects_df)
    c2.image(obejcts_image)
