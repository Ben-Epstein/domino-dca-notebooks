import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit import components


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
    st.title("Named Entity Recognition")
with row0_2:
    V_SPACE(1)
row3_spacer1, row3_1, row3_spacer2 = st.columns((0.1, 3.2, 0.1))
with row3_1:
    st.subheader("Enter a sentence and click extract to determine the organizations, locations, and people")
    V_SPACE(1)

#################
### SELECTION ###
#################

with st.form("my_form"):
    fintext = st.text_input(
        "Input text", "The Microsoft spokesperson Ronald Ramer lives in New York"
    )
    scored = st.form_submit_button("Extract")

results = list()

Please change this URL to where your model API has been deployed or load the model from where it has been stored

api_url = (
    "https://demo2.dominodatalab.com:443/models/65124411b03a930e975ea102/latest/model"
)

api_token = "VpPZFx2CrKooOgYuI44Nh60Lagxbig4UpN0pni0SDvLrKM6pIR1UNG9ExydoqBhc"

response = requests.post(
    api_url, auth=(api_token, api_token), json={"data": {"sentence": fintext}}
)
try:
    results.append(response.json().get('result'))
except requests.exceptions.RequestsJSONDecodeError:
    # Handle the error as you see fit.
    # For example, you can log the error and continue with default values.
    print(f"Failed to decode JSON from response")
    results.append("Failed to decode JSON from response, please check your model API")  # or provide a default value

# To test without a model endpoint, you can comment out the API code above and uncomment this
# from transformers import pipeline
# MODEL = "Domino-ai/distilbert-base-cased-wikiann"
# ner = pipeline(aggregation_strategy="simple", task="token-classification", model=MODEL)
# results.append(ner(fintext))


### Results ###
GREEN = '\033[92m'
BLUE = '\033[94m'
CYAN = '\033[96m'
ENDC = '\033[0m'

SPAN_MAP = {
    "ORG": "blue",
    "LOC": "green",
    "PER": "red"
}
result_text = "NA"

if results:
    result_text = fintext
    for token in results[0]:
        entity = token["entity_group"]
        span = token["word"]
        color = SPAN_MAP[entity]
        result_text = result_text.replace(span, f":{color}[{span}<sub>{entity}</sub>]")

#################
### SHOW RESULTS ###
#################

row4_spacer1, row4_1, row4_spacer2 = st.columns((0.2, 7.1, 0.2))
with row4_1:
    # st.subheader(result_text)
    st.markdown(f"## {result_text}", unsafe_allow_html=True)
    V_SPACE(1)
