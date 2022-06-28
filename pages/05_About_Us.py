from io import StringIO
from PIL import Image
from collections import Counter
import re, os, glob, copy, random, pickle, itertools,json
import ginza, spacy
import streamlit as st
import pandas as pd
import numpy as np
import boto3
from boto3.session import Session

# st.legacy_caching.clear_cache()
# homepageHolder.container()
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.empty()
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def aboutUs():
    c1,c2,c3=st.columns([1,6,1])
    with c2: st.image("./materials/indeed_logo_blue.png")
    st.markdown("<h1 style='text-align: center'>原稿解析・作成支援ツール</h1>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([1,6,1])
    with c2: st.image("./materials/AIRCreW_logo.PNG")
    st.markdown("<h2 style='text-align: center'><b>via</b></h2>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([1,6,1])
    with c2: st.image("./materials/VQlogo.png")

aboutUs()

try:
    del st.session_state["statTargetTxt"]
    del st.session_state["statBackgroundAvg"]
    del st.session_state["industry"]
    st.session_state["phase1DataLoaded"] = False
except KeyError:
    st.info("すでにリセットされております。")
