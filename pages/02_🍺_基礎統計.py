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
from boto3.dynamodb.conditions import Attr,Key
from decimal import Decimal

dynamodb = boto3.Session(profile_name="genikko-profile",).resource("dynamodb",region_name="ap-northeast-1",)
dynamoTableStats = dynamodb.Table("statistics_doc_indeed")

def docAveRec(df):

    dfStatMean = df.mean().tolist()[:-3]
    dfStatMean4Rec = ["dummy0","dummy1"] + dfStatMean + multiAddDiv(df)
    st.session_state.statBackgroundAvg = dfStatMean4Rec

    return dfStatMean4Rec

########### sidebar & settings 
st.set_page_config(
    page_title="AIRCreW - Fundamental Statistics",
    page_icon="./materials/favicon.ico",
    layout="centered",
    initial_sidebar_state="auto",
    )
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
sidebarDisplay(taskLabel="基礎統計")
funStaHolder = st.empty()
funStaContainer = funStaHolder.container()
########################################
st.empty()
funStaContainer.markdown(str_block_css,unsafe_allow_html=True)
funStaContainer.markdown("""
    <h1 style="text-align:start;">
        基礎<font color="deepskyblue">統計</font>量
            <sub class="pagetitle">&nbsp;<font color="deepskyblue">F</font>undamental <font color="deepskyblue">S</font>tatistics
            </sub></h1><hr>
    """,unsafe_allow_html=True)


########## いろいろなデータをロードするパート
if ("industry" not in st.session_state) or (st.session_state.industry == "-") or (st.session_state.industry == "notSelected"):
    st.error("業種を選択してください。")
    st.stop()

with st.spinner("対象原稿解析中..."):

    #if "phase1DataLoaded" not in st.session_state:
    #    st.session_state.phase1DataLoaded = False
    if ("statTargetTxt" not in st.session_state) or (st.session_state["statTargetTxt"] == "fileChanged"):
        try:
            targetTitle, targetContent = titleContent(st.session_state.target_file)
            st.session_state["targetTitle"] = targetTitle
            st.session_state["targetContent"] = targetContent
        except AttributeError:
            st.error("ファイルをアップロードしてください。")
            st.stop()

with st.spinner("上位原稿群解析中..."):

    if ("statTargetTxt" not in st.session_state) or (st.session_state["statTargetTxt"] == "fileChanged"):

        with st.spinner("データベース接続中..."):
            targetDocRec = statTargetDoc(targetTitle,targetContent)

        resStats = dynamoTableStats.scan()
        dataStats = resStats["Items"]
        while "LastEvaluatedKey" in dataStats:
            resStats = dynamoTableStats.scan(ExclusiveStartKey=resStats["LastEvaluatedKey"])
            dataStats.extend(resStats["Items"])
        dfSpecificField = pd.DataFrame(dataStats)
        # dfBackground = pd.read_csv(f"./rawData/prev_Stat_{st.session_state.industry}.csv")
        uidConditonTable = pd.read_csv("rawData/uidTitleConditon.csv")
        condition4DocSim = gyosyuKeywordDict[st.session_state.industry]

        dfSpecificUID = uidConditonTable.query('condition in @condition4DocSim')
        specificUID = dfSpecificUID["uid"].tolist()

        dfBackground = dfSpecificField.query(
            'uid in @specificUID'
            ).applymap(
                lambda x: float(str(x)) if type(x) == Decimal else x
                )
        dfBackground = dfBackground.reindex(
            columns=["uid",
                "タイトル字数","タイトル語数","タイトル名詞数",
                "原稿字数","原稿語数","原稿異なり語数","原稿名詞数","原稿異なり名詞数","原稿文数",
                "文平均字数","文平均語数","文平均名詞数"
                ])
        
        st.session_state.dfBackground = dfBackground
        lastRowRec = docAveRec(dfBackground)

    else:
        targetDocRec = st.session_state["statTargetTxt"]
        dfBackground = st.session_state["dfBackground"]
        lastRowRec = st.session_state["statBackgroundAvg"]

########## phase 1 part 1 統計データ表示パート
with funStaContainer:

    indexRange1 = [2,3,4,-3,-2,-1]
    indexRange2 = [5,6,7,8,9,10]
    labelRange = ["dummy1","dummy2","職種字数","職種語数","職種名詞数","原稿字数","原稿語数","原稿語数(異)","原稿名詞数","原稿名詞数(異)","原稿文数","文平均字数","文平均語数","文平均名詞数"]
    
    try:
        st.markdown(str_block_css,unsafe_allow_html=True)
        st.markdown(f"""
            <p>対象原稿職種：<span class="strblockGray">{st.session_state["targetTitle"]}</span></p>
            """, unsafe_allow_html=True,)
    except KeyError:
        st.info("原稿をアップロードしてください。")
        st.stop()
    for (i,col) in zip(indexRange1,st.columns(6)):
        targetNum = np.round(targetDocRec[i],decimals=1)
        deltaNum = np.round(targetDocRec[i]-lastRowRec[i],decimals=1)
        col.metric(labelRange[i],targetNum,deltaNum)

    st.markdown(str_block_css,unsafe_allow_html=True)
    st.markdown(f"""
        <p>対象原稿内容：<span class="strblockGray">{st.session_state["targetContent"][:25]}（以下略）</span></p>
        """, unsafe_allow_html=True,)
    for (i,col) in zip(indexRange2,st.columns(6)):
        targetNum = np.round(targetDocRec[i],decimals=1)
        deltaNum = np.round(targetDocRec[i]-lastRowRec[i],decimals=1)
        col.metric(labelRange[i],targetNum,deltaNum)

    st.markdown("<hr>", unsafe_allow_html=True)


########## phase 2 part 2 レーダーチャート表示パート
with funStaContainer:

    lastRowRec_title = lastRowRec[2:5] + lastRowRec[-3:]
    lastRowRec_content = lastRowRec[5:-3]
    targetDocRec_title = targetDocRec[2:5] + targetDocRec[-3:]
    targetDocRec_content = targetDocRec[5:-3]

    col1, col2 = st.columns(2)
    
    with col1:

        st.markdown("<h4 style='text-align: center; color: black;'>文単位解析</h4>", unsafe_allow_html=True)

        df4RadarChart_title = dfBackground[["タイトル字数","タイトル語数","タイトル名詞数","文平均字数","文平均語数","文平均名詞数"]]
        appendDF = pd.DataFrame(
            [lastRowRec_title,targetDocRec_title],
            columns=["タイトル字数","タイトル語数","タイトル名詞数","文平均字数","文平均語数","文平均名詞数"])
        df4RadarChart_title_med = df4RadarChart_title.append(appendDF)
        df4RadarChart_title_rank = df4RadarChart_title_med.rank(
            method="dense",
            ascending=True,
            pct=True)

        fig4RadarChart_title = radar_chart(
            df4RadarChart_title_rank,
            categoryRadarChart=["職種字数","職種語数","職種名詞数","平均字数","平均語数","平均名詞数"])
        st.plotly_chart(
            fig4RadarChart_title,
            use_container_width=True)

        with st.expander("データ",expanded=True):

            st.dataframe(
                data = df4RadarChart_title_med[["タイトル字数","タイトル語数","タイトル名詞数","文平均字数","文平均語数","文平均名詞数"]],
                width = 800,
                height = 400,)


    with col2:

        st.markdown("<h4 style='text-align: center; color: black;'>原稿単位解析</h4>", unsafe_allow_html=True)

        df4RadarChart_content = dfBackground[["原稿字数","原稿語数","原稿異なり語数","原稿名詞数","原稿異なり名詞数","原稿文数"]]
        appendDF = pd.DataFrame(
            [lastRowRec_content,targetDocRec_content],
            columns=["原稿字数","原稿語数","原稿異なり語数","原稿名詞数","原稿異なり名詞数","原稿文数"])
        df4RadarChart_content_med = df4RadarChart_content.append(appendDF)
        df4RadarChart_content_rank = df4RadarChart_content_med.rank(
            method="dense",
            ascending=True,
            pct=True)

        fig4RadarChart_content = radar_chart(
            df4RadarChart_content_rank,
            categoryRadarChart=["原稿字数","原稿語数","原稿異なり語数","原稿名詞数","原稿異なり名詞数","原稿文数"])
        st.plotly_chart(
            fig4RadarChart_content,
            use_container_width=True)

        with st.expander("データ",expanded=True): 

            st.dataframe(
                data = df4RadarChart_content_med[["原稿字数","原稿語数","原稿異なり語数","原稿名詞数","原稿異なり名詞数","原稿文数"]].astype(int),
                width = 800,
                height = 400,)  