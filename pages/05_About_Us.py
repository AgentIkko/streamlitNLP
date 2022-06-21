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

# st.empty()
# st.legacy_caching.clear_cache()
# homepageHolder.container()

c1,c2,c3=st.columns([1,6,1])
with c2: st.image("./indeed_logo_blue.png")
st.markdown("<h1 style='text-align: center'>原稿解析・作成支援ツール</h1>",unsafe_allow_html=True)
c1,c2,c3=st.columns([1,6,1])
with c2: st.image("./AIRCreW_logo.PNG")
st.markdown("<h2 style='text-align: center'><b>via</b></h2>",unsafe_allow_html=True)
c1,c2,c3=st.columns([1,6,1])
with c2: st.image("./VQlogo.png")