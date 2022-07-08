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
from boto3.dynamodb.conditions import Attr,Key
from decimal import Decimal

########### なぜか import できない func
dynamodb = boto3.Session(profile_name="genikko-profile",).resource("dynamodb",region_name="ap-northeast-1",)

def kwInTargetFile(_content):

    nounInContent = [e[0] for e in _content["info"] if e[1].startswith("名詞")]
    nounChunkInContent = list(itertools.chain.from_iterable([e.strip().split() for e in _content["info_nounChunk"]]))
    kwInContent = [e for e in list(set(nounInContent + nounChunkInContent)) if len(e) > 1]

    return kwInContent

def getDeviationValue(dfCol):
    seriesCal = dfCol.dropna()
    seriesCal_std = seriesCal.std(ddof=0)
    seriesCal_mean = seriesCal.mean()
    result = seriesCal.map(lambda x: round((x - seriesCal_mean) / seriesCal_std * 10 +50)).astype(int).tolist()
    return result

########################################

########### sidebar & settings
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.set_page_config(
    page_title="AIRCreW - Semantic Similarity",
    page_icon="./materials/favicon.ico",
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
        default=list(set(generalKeywords[:2] + selectedGyosyu[:3])),
        help="偏差値計算済みキーワード：正社員 | アルバイト | 飲食 | キッチンスタッフ | 調理補助 | シニア | 主婦 | 大学生",
        )
    additionalKeyWordInputForm = st.text_input(
        label="追加で調べたいキーワードを入力してください",
        value="",
        max_chars=64,
        placeholder="e.g. keyword1,keyword2,keyword3",
        help="計算時間の関係でDEMO時は入力しないでください",
        )

    srConfirmButton = st.form_submit_button("キーワード確定")

if not srConfirmButton:
    st.info("キーワードを選択してください。\n")
    st.stop()

# if srConfirmButton:
else:
    condition4DocSim = gyosyuKeywordDict[st.session_state.industry]
    dynamoTableRead = dynamodb.Table("jobposting_indeed")

    itemsFromDB = []
    for kw in condition4DocSim:
        res = dynamoTableRead.query(
            KeyConditionExpression=Key('condition').eq(kw),
            )
        itemsFromDB.extend(res['Items'])
        while "LastEvaluatedKey" in res:
            res = dynamoTableRead.query(
                KeyConditionExpression=Key('condition').eq(kw),
                ExclusiveStartKey=res["LastEvaluatedKey"],
                )
            itemsFromDB.extend(res['Items'])
    dfSpecificField = pd.DataFrame(itemsFromDB)

    # dataframe のやり方
    # prevEntDFWhole = pd.read_csv("./rawData/indeedPrev.csv")
    # dfSpecificField = prevEntDFWhole.query('condition in @condition4DocSim')

    candidateKeyWord = keyWordSelectForm
    if additionalKeyWordInputForm != "":
        candidateKeyWord += additionalKeyWordInputForm.split(",")
    candidateKeyWord = list(set([e for e in candidateKeyWord if len(e) > 0]))

    if "phase2candidateKW" not in st.session_state:
        st.session_state.phase2candidateKW = []
    st.session_state.phase2candidateKW = candidateKeyWord

    ########################################
    with st.spinner("Loading ..."):
        
        nlpedTarget = ginzaProcessing(task="singleText",sent1=txtContentSR)
        kwInTarget = kwInTargetFile(_content=nlpedTarget)

        if "similarityPairwise" not in st.session_state:

            st.session_state.similarityPairwise = {}

            dynoTableRead2 = dynamodb.Table("similarity_pairwise_keyword")
            resPairwiseKW = dynoTableRead2.scan()
            dataPairwiseKW = resPairwiseKW["Items"]
            while "LastEvaluatedKey" in resPairwiseKW:
                resPairwiseKW = dynoTableRead2.scan(ExclusiveStartKey=resPairwiseKW["LastEvaluatedKey"])
                dataPairwiseKW.extend(resPairwiseKW["Items"])
            prevSimDB = pd.DataFrame(dataPairwiseKW).set_index("Entity")
            st.session_state["similarityPairwise"] = prevSimDB

        st.success("Database loaded 1/2 ...")

        # with open("./rawData/prevPairwiseSimilarity.pkl","rb") as fr:
        #    prevDictSimScore4Entity = pickle.load(fr)
        # prevSimScore4Entity = pd.DataFrame.from_dict(prevDictSimScore4Entity)

        if "similarityDoc" not in st.session_state:

            st.session_state.similarityDoc = {}

            dynoTableRead3 = dynamodb.Table("similarity_doc_indeed")
            resDocSim = dynoTableRead3.scan()
            dataDocSim = resDocSim["Items"]
            while "LastEvaluatedKey" in resDocSim:
                resDocSim = dynoTableRead3.scan(ExclusiveStartKey=resDocSim["LastEvaluatedKey"])
                dataDocSim.extend(resDocSim["Items"])
            prevSimDocDB = pd.DataFrame(dataDocSim)#.set_index("uid")
            st.session_state["similarityDoc"] = prevSimDocDB

        st.success("Database loaded 2/2 ...")

##############################
prevSimScore4Entity = st.session_state["similarityPairwise"].applymap(lambda x: float(str(x)) if type(x) == Decimal else x)
prevSimScore4Doc = st.session_state["similarityDoc"].applymap(lambda x: float(str(x)) if type(x) == Decimal else x)

if "phase2TopEnt" not in st.session_state:
    st.session_state["phase2TopEnt"] = {}
st.session_state["phase2TopEnt"] = {kw:[] for kw in candidateKeyWord}

if "updatedDocSim" not in st.session_state:
    st.session_state["updatedDocSim"] = {}

for kw in candidateKeyWord:

    try:
        rankingDoc = prevSimScore4Doc[kw].tolist()
        clearedContent = dfSpecificField["jobContentCleared"]
        try:
            contraContentsKW = clearedContent.apply(eval).tolist()
        except TypeError:
            contraContentsKW = clearedContent.tolist()

    except KeyError:

        if kw in st.session_state["updatedDocSim"].keys():
            rankingDoc = st.session_state["updatedDocSim"][kw]

        else:
            contraUID = dfSpecificField["uid"].tolist()
            contraTitlesKW = dfSpecificField["jobTitle"].tolist()
            clearedContent = dfSpecificField["jobContentCleared"]
            try:
                contraContentsKW = clearedContent.apply(eval).tolist()
            except TypeError:
                contraContentsKW = clearedContent.tolist()

            prevSimScore4DocUpdate = pd.DataFrame(columns=["uid",kw])
            prevSimScore4DocUpdate["uid"] = dfSpecificField["uid"]

            barText = st.empty()
            myBar = st.progress(0)
            loopLength = len(contraUID)

            for barI,(u,t,c) in enumerate(zip(contraUID,contraTitlesKW,contraContentsKW)):

                docSimScoreContra = ginzaProcessing(
                    task = "pairText",
                    sent1 = kw,
                    sent2 = "\n".join(c),
                )["cosine_similarity"]

                prevSimScore4DocUpdate.loc[prevSimScore4DocUpdate["uid"]==u,kw] = docSimScoreContra

                barText.text(f"処理中　{round(((barI+1)/loopLength)*100,2)}%：{t[:26]}")
                myBar.progress(min((barI+1)/loopLength,100))

            rankingDoc = prevSimScore4DocUpdate[kw].tolist()
            st.session_state["updatedDocSim"][kw] = rankingDoc
            
            ##### db 書き込み
            dynamoTableUpdateDocSim = dynamodb.Table("similarity_doc_indeed")
            dfWrite = prevSimScore4DocUpdate.replace(to_replace=np.nan,value="").applymap(lambda x: Decimal(str(x)) if type(x) == float else x)
            for item in dfWrite.to_dict(orient="records"):
                options = {
                    "Key"                       :   {"uid"  :   item["uid"]},
                    "UpdateExpression"          :   "set #kw = :kw",
                    "ExpressionAttributeNames"  :   {"#kw"  :   kw},
                    "ExpressionAttributeValues" :   {":kw"  :   item[kw]},
                }
                dynamoTableUpdateDocSim.update_item(**options)

    ##### 対象原稿
    docSimScore = ginzaProcessing(
        task = "pairText",
        sent1 = kw,
        sent2 = txtContentSR,
    )["cosine_similarity"]

    rankingDoc.append(docSimScore)
    rankingDoc = pd.Series(rankingDoc).dropna()

    sss = rankingDoc.rank(
        ascending=True,
        pct=True).tolist()[-1] # targetを取ってきてるので tail
    sssDV = getDeviationValue(rankingDoc)[-1]

    expanderLabel = f"【{kw}】偏差値：{int(sssDV)}；順位：{round(sss*100,2)} %"

    ### keyword part
    # termlists = []
    # for content in contraContentsKW:
    #     termlist = ginzaProcessing(task = "singleText",sent1 = "\n".join(content))
    #     sNoun = [e[0].strip() for e in termlist["info"] if e[1].startswith("名詞")]
    #     termlists.extend(sNoun)
    # mostTerms = [e[0] for e in Counter(termlists).most_common() if (e[0] not in kwInTarget) and (kw not in e[0])]

    try:
        if prevSimScore4Entity[kw].max() == 0:
            termlists = []
            for content in contraContentsKW:
                termlist = ginzaProcessing(task = "singleText",sent1 = "\n".join(content))
                sNoun = [e[0].strip() for e in termlist["info"] if e[1].startswith("名詞")]
                termlists.extend(sNoun)
            entDisplay = [e[0] for e in Counter(termlists).most_common() if (e[0] not in kwInTarget) and (kw not in e[0])]
        else:
            sortedsim4EntData = prevSimScore4Entity.sort_values(by = kw, ascending = False,)
            entCandidate = sortedsim4EntData.index.values
            entDisplay = [e.strip() for e in entCandidate if (e not in kwInTarget) and (kw not in e)]

    except KeyError:
        st.info("データベースにないため、キーワード関連度計算中…")
        top18kterm = prevSimScore4Entity.index.values
        newScore4NewKW = []

        for term in top18kterm:

            if (len(newScore4NewKW)) > 20 and (max([e[1] for e in newScore4NewKW]) == 0):
                break

            term = term.strip()
            if (term not in kwInTarget) and (kw not in term):

                pairwiseSimScore = ginzaProcessing(
                    task = "pairText",
                    sent1 = kw,
                    sent2 = term,
                )["cosine_similarity"]

                decimalScore = Decimal(str(pairwiseSimScore))
                newScore4NewKW.append([term,decimalScore])

                options = {
                    "Key"                       :   {"Entity"  :   term},
                    "UpdateExpression"          :   "set #kw = :kw",
                    "ExpressionAttributeNames"  :   {"#kw"  :   kw},
                    "ExpressionAttributeValues" :   {":kw"  :   decimalScore},
                }
                dynoTableRead2.update_item(**options)

        if max([e[1] for e in newScore4NewKW]) == 0:
            termlists = []
            for content in contraContentsKW:
                termlist = ginzaProcessing(task = "singleText",sent1 = "\n".join(content))
                sNoun = [e[0].strip() for e in termlist["info"] if e[1].startswith("名詞")]
                termlists.extend(sNoun)
            entDisplay = [e[0] for e in Counter(termlists).most_common() if (e[0] not in kwInTarget) and (kw not in e[0])]
        else:
            entCandidate = sorted(newScore4NewKW,key=lambda x:x[1],reverse=True)
            entDisplay = [e[0] for e in entCandidate]

    st.session_state["phase2TopEnt"][kw] = entDisplay[:256]

    styledStr = "<p style='text-align:center;line-height:2.5;'>"
    for ent in entDisplay[:151]:
        styledStr += f"<span class='strblockBlue'>{ent}</span>"
    styledStr += "</p><hr>"
    with st.expander(expanderLabel):
        st.markdown(str_block_css,unsafe_allow_html=True)
        st.markdown(styledStr,unsafe_allow_html=True)