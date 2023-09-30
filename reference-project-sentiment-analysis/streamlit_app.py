import streamlit as st
from streamlit import components
import numpy as np
import pandas as pd
import requests


# helper function for blank streamlit lines
def V_SPACE(lines):
    for _ in range(lines):
        st.write("&nbsp;")


st.set_page_config(layout="wide")

####################
### INTRODUCTION ###
####################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 2.3, 0.1, 1.3, 0.1)
)
with row0_1:
    st.title("Amazon Review Sentiment Analysis")
with row0_2:
    V_SPACE(1)
row3_spacer1, row3_1, row3_spacer2 = st.columns((0.1, 3.2, 0.1))
with row3_1:
    st.subheader("Enter product review and click score to determine the sentiment")
    V_SPACE(1)

#################
### SELECTION ###
#################

with st.form("my_form"):
    fintext = st.text_input(
        "Input text", "The product is much smaller than expected or described."
    )
    scored = st.form_submit_button("Score")

results = list()

# Please change this URL to where your model API has been deployed or load the model from where it has been stored

api_url = (
    "https://demo2.dominodatalab.com:443/models/65124411b03a930e975ea102/latest/model"
)

api_token = "VpPZFx2CrKooOgYuI44Nh60Lagxbig4UpN0pni0SDvLrKM6pIR1UNG9ExydoqBhc"

response = requests.post(
    api_url, auth=(api_token, api_token), json={"data": {"sentence": fintext}}
)
results.append(response.json().get("result"))

### Results ###
result_text = "NA"
result_prob = -1

if results:
    result_text = results[0]["label"]
    result_prob = round(results[0]["score"], 4)

#################
### SHOW RESULTS ###
#################

row4_spacer1, row4_1, row4_spacer2 = st.columns((0.2, 7.1, 0.2))
with row4_1:
    st.subheader(
        f"The sentiment of this review is {result_text} with a probability of {result_prob}"
    )
    V_SPACE(1)
