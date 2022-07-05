from curses import keyname
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
import plotly.express as px
import plotly.graph_objects as go
from Home_Page import *

########### 何故か読み込まれない css style
css_indeedMimic = f"""
    <style>
        span.strblockIndeed{{
            padding         :   6px;
            border          :   1.5px solid gray;
            border-radius   :   8px;
            /*background      :   lightskyblue;*/
            font-size       :   70%;
            font-weight     :   bold;
            margin          :   5px;
            display         :   inline-block;
        }}
    </style>
    """
########### sidebar & settings 
st.set_page_config(
    page_title="AIRCreW - Data Scraping",
    page_icon="./materials/favicon.ico",
    layout="centered",
    initial_sidebar_state="auto",
    )
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
sidebarDisplay(taskLabel="原稿抽出")
datScrHolder = st.empty()
datScrContainer = datScrHolder.container()
########################################

########### 何故か動かないfunc
def atsScraping(url,ats_Type):

    payload = {
        "url"   :   url,
        "atsType"   :   atsType,
        }
    
    response = clientLambda.invoke(
        FunctionName = 'arn:aws:lambda:ap-northeast-1:836835325374:function:bs4scraping',
        InvocationType = 'RequestResponse',
        Payload = json.dumps(payload),
    )

    result = json.loads(response["Payload"].read().decode("utf-8"))

    return result

def scrapedData2TargetFile(dicStored):

    outputText = dicStored["detailTitle"] + "\n\n"

    for e in dicStored.keys():
        if e not in ["detailTitle","images"]:
            part = dicStored[e]
            if isinstance(part,list):
                outputText += "\n".join(part) + "\n"
            elif isinstance(part,str):
                outputText += part + "\n"
    
    return outputText.strip()


        
########################################

st.empty()
datScrContainer.markdown(str_block_css,unsafe_allow_html=True)
datScrContainer.markdown("""
    <h1 style="text-align:start;">
        原稿<font color="deepskyblue">抽出</font>
            <sub class="pagetitle">&nbsp;<font color="deepskyblue">D</font>ata <font color="deepskyblue">S</font>craping
            </sub></h1>
    """,unsafe_allow_html=True)

########## url input & type select

with datScrContainer.form("dataScraping4specificATS"):

    inputURL = st.text_input(
        label="求人ページURLを貼り付けてください",
        max_chars=128,
        )
    atsType = st.radio(
        label="ATSの種類を選択してください",
        options=["自社","リクオプ","ジョブオプ","Airワーク"],
        horizontal=True)

    optionGyosyu = st.selectbox(
        label="業種を選択してください", 
        options=["-"] + list(gyosyuKeywordDict.keys()),
        )
    gyosyuStore = optionGyosyu



    startButton = st.form_submit_button(label="抽出開始")

########################################

if startButton:

    if "url" not in st.session_state:
        st.session_state["url"] = inputURL

    elif "url" in st.session_state:
        if st.session_state["url"] == inputURL:
            st.info("スクレイピングされたデータ利用中")
        elif st.session_state["url"] != inputURL:
            st.session_state["url"] = inputURL
            st.session_state["statTargetTxt"] = "fileChanged"

    try:
        scrapedPosting = atsScraping(url = st.session_state.url,ats_Type=atsType)
    except Exception:
        st.error("エラー発生中。")
        st.stop()

    st.session_state["scrapedItem"] = scrapedPosting

    if "industry" not in st.session_state:
        st.session_state.industry = "notSelected"
    st.session_state.industry = gyosyuStore

########################################

########## なぜか読み込めない func (display part)

def indeedFormat():
    # companyName = st.session_state["detailTitle"].split("／")[-1]+"\n"+st.session_state["location1"]
    companyName = st.session_state["scrapedItem"]["location1"]
    # locationModi = st.session_state["location3"].split("／")[-1].strip()
    stationModi = st.session_state["scrapedItem"]["station"]
    locationModi = st.session_state["scrapedItem"]["location3"]
    # salaryModi = st.session_state["salary"].split(" ")[0]+"\n"+" ".join(st.session_state["salary"].split(" ")[1:])
    salaryModi = st.session_state["scrapedItem"]["salary"]
    jobModi = st.session_state["scrapedItem"]["job"]

    tokucho = "<p style='text-align:start;line-height:1;margin:0.5em 0px;'>"
    for e in st.session_state["scrapedItem"]["conditions"]:
        tokucho += f"""
            <span style='padding:4px;border:0.5px solid gray;border-radius:2px;font-size:65%;font-weight:bold;margin:0.5em 1px;display:inline-block;'>
            {e}</span>"""
    tokucho += "</p>"

    indeedMimic.markdown(f"""
        <p style='text-align:start;line-height:1;margin:0.5em 0px;'>
        <span style='padding:4px;border:0px;border-radius:2px;font-size:65%;font-weight:bold;background:whitesmoke;margin:0.5em 1px;display:inline-block;'>
        {st.session_state["scrapedItem"]["status"]}
        </span></p>
        """,
        unsafe_allow_html=True)

    indeedMimic.markdown(f"""
        <h5>{st.session_state["scrapedItem"]["detailTitle"]}</h5>""",
        unsafe_allow_html=True)

    indeedMimic.markdown(f"""
        <p style='text-align:center;line-height:2;margin:0.5em 0px;background:whitesmoke;border-block:0.5px solid;'>
        <span style='padding:4px;border:0px;writing-mode:horizontal-tb;border-radius:2px;font-size:65%;font-weight:bold;margin:0.5em 1px;'>
        [キャッチコピー]
        </span></p>
        """,
        unsafe_allow_html=True)

    indeedMimic.markdown(f"""
        <p style="font-size:70%; font-weight:bold; margin:0.5em 0px;">会社名</p>
        <p style="font-size:70%; margin:0.5em 0px;">{companyName}</p>
        <p style="font-size:70%; font-weight:bold; margin:0.5em 0px;">勤務地</p>
        <p style="font-size:70%; margin:0.5em 0px;">{stationModi}</p>
        <p style="font-size:70%; margin:0.5em 0px;">{locationModi}</p>
        <p style="font-size:70%; font-weight:bold; margin:0.5em 0px;">給与</p>
        <p style="font-size:70%; margin:0.5em 0px;">{salaryModi}</p>
        <p style="font-size:70%; font-weight:bold; margin:0.5em 0px;">職種</p>
        <p style="font-size:70%; margin:0.5em 0px;">{jobModi}</p>
        <p style="font-size:70%; font-weight:bold; margin:0.5em 0px;">仕事の特徴</p>
        {tokucho}
        """,
        unsafe_allow_html=True)

    workContentStr = "</p><p style='font-size:10pt;margin:0 0 0.5rem;'>".join(st.session_state["scrapedItem"]["workContent"].split("\n"))
    detailTextStr = "</p><p style='font-size:10pt;margin:0 0 0.5rem;'>".join(st.session_state["scrapedItem"]["detailText"].split("\n"))

    indeedMimic.markdown(f"""
        <b>仕事内容</b>　<del>　　　　　　　　　　　　　　　　</del>
        <div>
        <p style="font-size:10pt;">{workContentStr}</p>
        <p style="font-size:10pt;">{detailTextStr}</p>
        </div>
        """,
        unsafe_allow_html=True)

    indeedMimic.markdown(f"""
        <b>勤務時間・休日</b>　<del>　　　　　　　　　　　　　</del>
        <div>
        <p style="font-size:10pt;">
        {st.session_state["scrapedItem"]["shiftTime2"]}
        </p>
        <p style="font-size:10pt;">
        {st.session_state["scrapedItem"]["duration"]}
        </p></div>
        """,
        unsafe_allow_html=True)

    indeedMimic.markdown(f"""
        <b>勤務地</b>　<del>　　　　　　　　　　　　　　　　　</del>
        <p style="font-size:10pt;">
        {st.session_state["scrapedItem"]["location2"]}
        </p>
        <p style="font-size:10pt;">
        {st.session_state["scrapedItem"]["location3"]}
        </p>
        """,
        unsafe_allow_html=True)

    indeedMimic.markdown(f"""
        <b>給与・待遇</b>　<del>　　　　　　　　　　　　　　　</del>
        <div>
        <p style="font-size:10pt;">
        {st.session_state["scrapedItem"]["qualification"]}
        </p>
        <p style="font-size:10pt;">
        {st.session_state["scrapedItem"]["wellBeing"]}
        </p>
        </div> 
        """,
        unsafe_allow_html=True)

    indeedMimic.markdown(f"""
        <b>応募要項</b>　<del>　　　　　　　　　　　　　　　　</del>
        <div>
        <p style="font-weight:bold;font-size:10pt;">
        {st.session_state["scrapedItem"]["tel"]}
        <p></div>
        <div>
        <p style="font-size:10pt;">
        {'</p><p style="font-size:10pt;">'.join(st.session_state["scrapedItem"]["cautionInfo"])}
        </p></div>
        """,
        unsafe_allow_html=True)

    indeedMimic.markdown(f"""
        <b>その他</b>　<del>　　　　　　　　　　　　　　　　　</del>
        <div>
        <p style="font-size:10pt; font-weight:bold;">
        {st.session_state["scrapedItem"]['companyInfo1Title']}
        </p>
        <p style="font-size:10pt;">
        {st.session_state["scrapedItem"]['companyInfo1']}
        </p>
        <p style="font-size:10pt;font-weight:bold;">
        {st.session_state["scrapedItem"]['companyInfo2Title']}
        </p>
        <p style="font-size:10pt;">
        {st.session_state["scrapedItem"]['companyInfo2']}
        </p>
        </div>
        """,
        unsafe_allow_html=True)


########################################

########## rulebook for checklist (data preparation)
ruleBook = {
    "rule-00"   :   "画像3枚以上",
    "rule-01"   :   "単独の雇用形態",
    "rule-02"   :   "タイトル2行以内",
    "rule-03"   :   "タイトル：[施設]の[職種]",
    "rule-03-1" :   "単独の職種",
    "rule-04"   :   "駅表記",
    "rule-05"   :   "住所表記",
    "rule-06"   :   "単独の勤務地",
    "rule-07"   :   "勤務地の詳細表記",
    # "rule-08"   :   "広域エリア",
    "rule-09"   :   "必要経験・スキル表記",
    "rule-10"   :   "給与表記（「時給」等）",
    "rule-11"   :   "最低賃金以上",
    "rule-12"   :   "給与レンジ表記",
    "rule-13"   :   "特徴付ける表現の強調",
    "rule-14"   :   "誤植なし",
    "rule-15"   :   "会社情報",
    "rule-99"   :   "キーワードスタッフィングなし",
}

if "ruleBook" not in st.session_state:
    st.session_state["ruleBook"] = {}
st.session_state["ruleBook"] = ruleBook

########################################

########## 表示パート image
try:
    imageLen = len(st.session_state["scrapedItem"]["images"])
except KeyError:
    st.info("URLを入力して原稿を抽出してください。")
    st.stop()

for i,col in enumerate(st.columns(imageLen)):
    with col: st.image(st.session_state["scrapedItem"]["images"][i])

########## 表示パート
col1, col2 = st.columns(2)

try:
    with col1:

        if "target_file" not in st.session_state:
            st.session_state["target_file"] = ""
        st.session_state["target_file"] = scrapedData2TargetFile(dicStored=st.session_state["scrapedItem"])

        indeedMimic = st.container()
        indeedFormat()

    with col2:

        with st.form("check4MimicPosting"):

            for k,v in st.session_state["ruleBook"].items():
                ruleFlag = st.checkbox(label=v,key=k)

            ruleSubmit = st.form_submit_button(label="Check !")
        
        if ruleSubmit:
            ruleFlagList = [e for e in st.session_state if e.startswith("rule-")]
            for r in ruleFlagList:
                if not st.session_state[r]:
                    st.write(st.session_state["ruleBook"][r])

except KeyError:
    st.info("URLを入力して原稿を抽出してください。")
    st.stop()