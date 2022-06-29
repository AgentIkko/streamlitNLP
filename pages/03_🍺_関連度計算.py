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

########### なぜか import できない func

def kwInTargetFile(_content):

    nounInContent = [e[0] for e in _content["info"] if e[1].startswith("名詞")]
    nounChunkInContent = list(itertools.chain.from_iterable([e.strip().split() for e in _content["info_nounChunk"]]))
    kwInContent = [e for e in list(set(nounInContent + nounChunkInContent)) if len(e) > 1]

    return kwInContent

########################################

########### sidebar & settings
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.set_page_config(
    page_title="AIRCreW - Semantic Similarity",
    page_icon="./favicon.ico",
    layout="centered",
    initial_sidebar_state="auto",
    )
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
sidebarDisplay(taskLabel="関連度計算")
semSimHolder = st.empty()
semSimContainer = semSimHolder.container()
########################################
st.empty()
semSimContainer.markdown(str_block_css,unsafe_allow_html=True)
semSimContainer.markdown("""
    <h1 style="text-align:start;">
        キーワード<font color="deepskyblue">関連</font>度
            <sub class="pagetitle">&nbsp;<font color="deepskyblue">S</font>emantic <font color="deepskyblue">S</font>imilarity
            </sub></h1><hr>
    """,unsafe_allow_html=True)

try:
    txtTitleSR, txtContentSR = titleContent(st.session_state.target_file)
except AttributeError:
    st.error("ファイルをアップロードしてください。")
    st.stop()

########## 調べたいキーワードを確定するパート
if ("industry" not in st.session_state) or (st.session_state.industry == "-") or (st.session_state.industry == "notSelected"):
    st.error("業種を選択してください。")
    st.stop()
    
for kw in gyosyuKeywordDict.keys():
    if st.session_state.industry == kw:
        selectedGyosyu = gyosyuKeywordDict[kw]
    elif st.session_state.industry == "-":
        selectedGyosyu = generalKeywords[:5]

optionKeywords = list(set(generalKeywords + selectedGyosyu))

with st.form("phase2_KeywordSelect"):

    st.markdown(str_block_css,unsafe_allow_html=True)
    keyWordSelectForm = st.multiselect(
        label="キーワード候補",
        options=optionKeywords,
        default=list(set(generalKeywords[:2] + selectedGyosyu[:2])),
        help="DEMO用キーワード：介護, 正社員",
        )
    additionalKeyWordInputForm = st.text_input(
        label="追加で調べたいキーワードを入力してください",
        value="",
        max_chars=64,
        placeholder="e.g. keyword1,keyword2,keyword3",
        help="DEMO時は入力しないでください",
        )

    srConfirmButton = st.form_submit_button("キーワード確定")

if srConfirmButton:

    candidateKeyWord = keyWordSelectForm
    if additionalKeyWordInputForm != "":
        candidateKeyWord += additionalKeyWordInputForm.split(",")
    candidateKeyWord = list(set([e for e in candidateKeyWord if len(e) > 0]))

    if "phase2candidateKW" not in st.session_state:
        st.session_state.phase2candidateKW = []
    st.session_state.phase2candidateKW = candidateKeyWord

    ########################################

    nlpedTarget = ginzaProcessing(task="singleText",sent1=txtContentSR)
    kwInTarget = kwInTargetFile(_content=nlpedTarget)

    
    try:
        # 修正必要
        prevSimScore4Entity = pd.read_csv(f"phase2EntityIn{st.session_state.industry}.csv",index_col="keyword").T
        # st.dataframe(prevSimScore4Entity)
        # prevSimScore4Entity = pd.read_csv(f"phase2_Ent_{st.session_state.industry}.csv")

        st.warning("CAUTION: 計算済みデータを利用しております。")

    except Exception:
        with st.spinner("キーワードの関連度を計算しております。少々お待ちください。"):
            ##########
            st.info("過去原稿を解析中...")
            prevEntDF = pd.read_csv(f"{st.session_state.industry}_sponsorPro_text.csv")

            # 修正必要
            prevSentCorpus = []
            for e in prevEntDF["jobDescriptionText"].tolist():
                eList = e.strip().split("\n")
                for ee in eList:
                    eeStr = noSymbolic(ee).strip()
                    if len(eeStr) > 0:
                        prevSentCorpus.append(eeStr)

            allEntNC = []
            st.write(len(prevSentCorpus))

            for sentence in prevSentCorpus:
                res = ginzaProcessing(task="singleText",sent1=sentence)
                resEntNC = kwInTargetFile(res)
                allEntNC += resEntNC
            
            allEntNC = list(set(allEntNC))[:1000]

            dictOfSim4Entity = {ent : [] for ent in allEntNC}
            dictOfSim4Entity.update({"keyword": []})
            dictOfSim4Entity["keyword"] += candidateKeyWord

            for ent in allEntNC:
                for kw in candidateKeyWord:
                    sim4Ent = ginzaProcessing(task="pairText",sent1=ent,sent2=kw)["cosine_similarity"]
                    dictOfSim4Entity[ent].append(sim4Ent)

            sim4EntData = pd.DataFrame.from_dict(dictOfSim4Entity)
            sim4EntData.to_csv(f"phase2EntityIn{st.session_state.industry}.csv")
            st.success("解析完了🎈")
            ##########

    try:
        simScoreData = pd.read_csv(f"phase2_Doc_{st.session_state.industry}.csv") 
    except Exception:
        ########## 関連度計算
        st.info("原稿順位計算中...")
        dictOfSimScores = {kw: [] for kw in candidateKeyWord}
        dictOfSimScores.update({"職種":[]})

        dictOfSimScores["職種"].append("TARGET")

        for kw in candidateKeyWord:
            docSimScore = ginzaProcessing(
                task = "pairText",
                sent1 = kw,
                sent2 = txtContentSR,
            )["cosine_similarity"]
            dictOfSimScores[kw].append(docSimScore)

        st.success("対象原稿処理終了")

        dfSponsorProGenkou = pd.read_csv(f"{st.session_state.industry}_sponsorPro_text.csv")
        contraTitles = dfSponsorProGenkou["jobTitle"].tolist()
        contraContents = dfSponsorProGenkou["jobDescriptionText"].tolist()

        #status_text = st.empty()
        #loadBarSR = st.progress(0)
        #loopCount = len(contraTitles)

        for (t,c) in zip(contraTitles,contraContents):

            dictOfSimScores["職種"].append(t)

            for kw in candidateKeyWord:
                docSimScoreContra = ginzaProcessing(
                    task = "pairText",
                    sent1 = kw,
                    sent2 = noSymbolic(c),
                )["cosine_similarity"]
                dictOfSimScores[kw].append(docSimScoreContra)

        simScoreData = pd.DataFrame.from_dict(dictOfSimScores)
        simScoreData.to_csv(f"phase2_Ent_{st.session_state.industry}.csv")

##############################

# lambda のほうで ent の出力を追加
def getDeviationValue(df,colName):
    seriesCal = df[colName]
    seriesCal_std = seriesCal.std(ddof=0)
    seriesCal_mean = seriesCal.mean()
    result = seriesCal.map(lambda x: round((x - seriesCal_mean) / seriesCal_std * 10 +50)).astype(int).tolist()
    return result

if not srConfirmButton:
#if (not candidateKeyWord) or (candidateKeyWord == []):
    st.info("キーワードを選択してください。\n")
    st.stop()

for kw in candidateKeyWord:
    sss = simScoreData[kw].rank(
        ascending=True,
        pct=True).tolist()[0]
    sssDV = getDeviationValue(simScoreData,kw)[0]

    expanderLabel = f"【{kw}】偏差値：{int(sssDV)}；順位：{round(sss,4)*100} %"

    #st.metric("順位",sss)
    sortedsim4EntData = prevSimScore4Entity.sort_values(
        by = kw, ascending = False,)
    # entCandidate = sortedsim4EntData["NameEntity"].tolist()
    entCandidate = sortedsim4EntData.index.values
    entDisplay = [e.strip() for e in entCandidate if e not in kwInTarget]
    kwParsed = ginzaProcessing(task="singleText",sent1=kw)
    entDisplay = [e for e in entDisplay if kw not in e]

    with open(f"{kw}_save4now.pk","wb") as pklw:
        pickle.dump(entDisplay,pklw)

    styledStr = "<p style='text-align:center;line-height:2.5;'>"
    for ent in entDisplay[:101]:
        styledStr += f"<span class='strblockBlue'>{ent}</span>"
    styledStr += "</p><hr>"
    with st.expander(expanderLabel):
        st.markdown(str_block_css,unsafe_allow_html=True)
        st.markdown(styledStr,unsafe_allow_html=True)

            
