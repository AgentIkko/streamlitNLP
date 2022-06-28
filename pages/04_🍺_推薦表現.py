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
from Home_Page import *

########### sidebar & settings
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.set_page_config(
    page_title="AIRCreW - Recommended Expression",
    page_icon="./favicon.ico",
    layout="centered",
    initial_sidebar_state="auto",
    )
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
sidebarDisplay(taskLabel="推薦表現")
recExpHolder = st.empty()
recExpContainer = recExpHolder.container()
############################################

########### なぜか効かないfunc
def loadCorpus(corpath):
    dfKaigoSponsor = pd.read_csv(corpath)
    kaigoSponsorSentList = list(
        itertools.chain(
            *[article.split("\n") for article in dfKaigoSponsor.jobDescriptionText.tolist()]
            ))
    kaigoSponsorSentSet = [noSymbolic(e.strip()) for e in set(kaigoSponsorSentList) if len(e) > 0]
    kaigoSponsorSentSet = [e for e in list(set(kaigoSponsorSentSet)) if len(e.strip()) > 0]
    return kaigoSponsorSentSet

############################################
st.empty()    
recExpContainer.markdown(str_block_css,unsafe_allow_html=True)
recExpContainer.markdown("""
    <h1 style="text-align:start;">
        原稿<font color="deepskyblue">推薦</font>表現
            <sub class="pagetitle">&nbsp;<font color="deepskyblue">R</font>ecommended <font color="deepskyblue">E</font>xpression
            </sub></h1><hr>
    """,unsafe_allow_html=True)


if ("industry" not in st.session_state) or (st.session_state.industry == "-") or (st.session_state.industry == "notSelected"):
    st.error("業種を選択してください。")
    st.stop()

gyoSyuSents = loadCorpus(f"{st.session_state.industry}_sponsor_text.csv")

with open("介護スタッフ_save4now.pk","rb") as pklr:
    termlist = pickle.load(pklr)

phase3Candidate = termlist[:51]

with st.form("phase3"):
    phase3Form = st.multiselect(
        label="これらの関連キーワードでよろしいですか",
        options = phase3Candidate,
        default = phase3Candidate[:11],
        help = "数量やグループ等の調整が可能",
    )
    phase3ConfirmButton = st.form_submit_button("関連語確定")

phase3Term = phase3Form
if phase3ConfirmButton:
    for t in phase3Term:
        with st.expander(f"{t}を含む表現"):

            para4Display = ""

            for s in gyoSyuSents:
                if t in s:
                    sent4Display = f"<div class='parblock'>{s}</div><p></p>"
                    para4Display += sent4Display
            
            para4Display +="<hr>"
            st.markdown(str_block_css,unsafe_allow_html=True)
            st.markdown(para4Display,unsafe_allow_html=True)
