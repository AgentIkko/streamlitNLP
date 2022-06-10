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

#################### html wrapper & page config
session = Session(profile_name="genikko-profile")
client = session.client("sagemaker-runtime",region_name="ap-northeast-1")
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
str_block_css = f"""
    <style>
        span.strblock{{
            padding         :   8px;
            /*border          :   2px solid #666666;*/
            border-radius   :   10px;
            background      :   deepskyblue;
            font-weight     :   bold;
        }}
        div.parblock{{
            padding         :   8px;
            /*border          :   1px solid lightskyblue;*/
            border-radius   :   10px;
            background      :   #f3f4f7;
        }}
        sub.pagetitle{{
            font-size       :   50%;
            /*vertical-align  :   sub;*/
            font-variant    :   small-caps;
        }}
        span[data-baseweb="tag"]{{
            background-color:   deepskyblue;
        }}
    </style>
    """ 
hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
nlp = spacy.load("ja_ginza")
# streamlit では """ が効かない
st.set_page_config(
    page_title="原稿解析 AIRCreW",
    page_icon="./favicon.ico",
    layout="centered",
    initial_sidebar_state="auto",
    )

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

#################### button status
@st.cache(allow_output_mutation=True)
def button_states():
    return {"pressed":None}
def session_change():
    if "isPressedModel" in st.session_state:
        st.session_state["isPressedModel"].update({"pressed":None})
    if "isPressedTask" in st.session_state:
        st.session_state["isPressedTask"].update({"pressed":None})
    if "isPressedSR" in st.session_state:
        st.session_state["isPressedSR"].update({"pressed":None})
    
#################### sidebarImage
st.sidebar.image("./VQlogo.png")
st.sidebar.image(image="./AIRCreW_logo.PNG",width=300)

#################### homepage

homepageHolder = st.empty()
aboutusHolder = st.empty()
funStaHolder = st.empty()
semRelHolder = st.empty()
expRecHolder = st.empty()
jobGenHolder =st.empty()

infoPages = st.sidebar.radio("",options=(["HOME","ABOUT US"]))

if infoPages == "ABOUT US":
    funStaHolder.empty()
    st.legacy_caching.clear_cache()
    homepageHolder.container()

    with homepageHolder.container():
    
        c1,c2,c3=st.columns([1,6,1])
        with c2: st.image("./indeed_logo_blue.png")
        st.markdown("<h1 style='text-align: center'>原稿解析・作成支援ツール</h1>",unsafe_allow_html=True)
        c1,c2,c3=st.columns([1,6,1])
        with c2: st.image("./AIRCreW_logo.PNG")
        st.markdown("<h2 style='text-align: center'><b>via</b></h2>",unsafe_allow_html=True)
        c1,c2,c3=st.columns([1,6,1])
        with c2: st.image("./VQlogo.png")

if infoPages == "HOME":
    homepageHolder.empty()

#################### side bar
st.sidebar.markdown('<h4 style="text-align: center">ファイルをアップロード</h4>',unsafe_allow_html=True)

########## file upload
with st.sidebar.form("uploaderSingleFile",clear_on_submit=False):

    ## 1. from uploaded files
    fileUploaderForm = st.file_uploader(label="① 原稿テキストファイル")
    submitted = st.form_submit_button("アップロード")

    if fileUploaderForm and submitted is not None:
        st.write("ファイルをアップロードしました。")


########## 業種選択
optionGyosyu = st.sidebar.selectbox(label="STEP 1: 業種選択", options=("-","介護","物流","販売"))
st.sidebar.markdown(str_block_css,unsafe_allow_html=True)
st.sidebar.markdown(f"""
    <p style="text-align: center">現在の業種：<span class="strblock"><font color="white">{optionGyosyu}</font></span></p>
    """, unsafe_allow_html=True,
    )
    

########## タスク選択
optionPhase = st.sidebar.selectbox(label="STEP 2: タスク選択",options=("-","基礎統計","関連度計算","原稿推薦表現","原稿生成β"))
st.sidebar.markdown(str_block_css,unsafe_allow_html=True)
st.sidebar.markdown(f"""
    <p style="text-align: center">現在のタスク：<span class="strblock"><font color="white">{optionPhase}</font></span></p>
    """, unsafe_allow_html=True,
    )

########## info expander
with st.sidebar.expander("このアプリについて"):
    st.write("""
        AI(人工知能)のチカラを使ってindeed原稿を解析・改善する独自ツールを展開しています。
        原稿の比較検証等を可視化することで、改善ポイントを明確にして最適なindeed原稿へと導くことが可能です。
    """)

########## copyright
st.sidebar.markdown('<span style="font-size:12px;text-align: center"><b>Copyright ©2022 株式会社一広バリュークエスト</b></span>',unsafe_allow_html=True)

#################### FSセクション用関数
def multiAddDiv(df):
    result1 = sum(df["原稿文数"]*df["文平均字数"])/sum(df["原稿文数"])
    result2 = sum(df["原稿文数"]*df["文平均語数"])/sum(df["原稿文数"])
    result3 = sum(df["原稿文数"]*df["文平均名詞数"])/sum(df["原稿文数"])
    return [result1,result2,result3]

@st.cache
def readUploadedFile():

    if fileUploaderForm is not None:
        txt = StringIO(fileUploaderForm.getvalue().decode("utf-8")).read()
    else:
        with open("./testIkko.txt",encoding="utf-8") as fr:
            txt = fr.read()
            #txt = """一広バリュークエストの独自サービス\n
            #AIRCreWで原稿解析。\n
            #一広バリュークエストに運用をまかせると求人費用30%以上削減できる。"""
    
    genkouTitle = txt.split("\n")[0]
    genkouContent = "\n".join([e.strip() for e in txt.split("\n")[1:] if len(e)>0])

    return genkouTitle, genkouContent

@st.cache
def readDF(pathDF):

    dfStat = pd.read_csv(pathDF)
    dfStatMean = dfStat.mean().tolist()[1:]
    dfStatMean4Rec = ["x","x"] + dfStatMean[:-3] + multiAddDiv(dfStat)

    return dfStatMean4Rec

def forSentence(s):

    s = s.strip()
    sNLP = nlp(s)
    sWakati = [token.text for token in sNLP if not token.tag_.startswith("補助記号")]
    sNoun = [token.text for token in sNLP if token.tag_.startswith("名詞")]

    return [len(s),len(sWakati),len(sNoun)]

def forDescrption(des):

    des = des.strip()
    sentlist = [sent for sent in des.split("\n") if len(sent) > 0]
    desNLP = nlp(des)
    desWakati = [token.text for token in desNLP if not token.tag_.startswith("補助記号")]
    desNoun = [token.text for token in desNLP if token.tag_.startswith("名詞")]
    resultsA = [len(des),len(desWakati),len(set(desWakati)),len(desNoun),len(set(desNoun)),len(sentlist),]
    resultsASent = []

    for sent in sentlist:
        resultsASent.append(forSentence(sent))
    sMean = np.mean(resultsASent, axis=0)

    results = resultsA + sMean.tolist()
    
    return results

def getDeviationValue(df,colName):

    seriesCal = df[colName]
    seriesCal_std = seriesCal.std(ddof=0)
    seriesCal_mean = seriesCal.mean()
    result = seriesCal.map(lambda x: round((x - seriesCal_mean) / seriesCal_std * 10 +50)).astype(int).tolist()

    return result

#################### 基礎統計セクション
if optionPhase == "基礎統計":

    import plotly.express as px
    import plotly.graph_objects as go

    homepageHolder.empty()
    funStaContainer = funStaHolder.container()
    funStaContainer.markdown(str_block_css,unsafe_allow_html=True)
    funStaContainer.markdown("""
        <h1 style="text-align:start;">
            基礎<font color="deepskyblue">統計</font>量
                <sub class="pagetitle">&nbsp;<font color="deepskyblue">F</font>undamental <font color="deepskyblue">S</font>tatistics
                </sub></h1>
        """,unsafe_allow_html=True)

    funStaContainer.write("HELLO")

#################### SR
if optionPhase == "関連度計算":
    
    homepageHolder.empty()
    semRelContainer = semRelHolder.container()
    semRelContainer.markdown(str_block_css,unsafe_allow_html=True)
    semRelContainer.markdown("""
        <h1 style="text-align:start;">
            キーワード<font color="deepskyblue">関連</font>度
                <sub class="pagetitle">&nbsp;<font color="deepskyblue">S</font>emantic <font color="deepskyblue">S</font>imilarity
                </sub></h1>
        """,unsafe_allow_html=True)
    txtTitleSR, txtContentSR = readUploadedFile()

    generalKeywords = [
        "正社員","アルバイト","中高年","主婦パート","土日祝休み",
        "シニア","ハローワーク","深夜バイト","オープニングスタッフ",
        ]
    
    gyosyuKeywordDict = {
        "介護":[
            "介護","ホームヘルパー","介護職員","介護スタッフ",
            "福祉","生活支援員","デイサービス","訪問介護","グループホーム","介護職",
            "世話人","社会福祉士","夜勤専従","特別養護老人ホーム","就労支援員",
            ],
        "物流":[
            "物流","輸送",
            ],
        "販売":[
            "アパレル販売","販売スタッフ",
            ],
        "営業":[
            "営業","不動産",
            ],
        }

    for kw in gyosyuKeywordDict.keys():
        if optionGyosyu == kw:
            selectedGyosyu = gyosyuKeywordDict[kw]
        elif optionGyosyu == "-":
            selectedGyosyu = generalKeywords[:5]

    optionKeywords = list(set(generalKeywords + selectedGyosyu))

    ########## 関連度を調べたいキーワードの選択と自由入力
    with st.form("keywordselect"):
        st.markdown(str_block_css,unsafe_allow_html=True)
        keyWordSelectForm = st.multiselect(
            label="キーワード",
            options=optionKeywords,
            default=list(set(generalKeywords[:3] + selectedGyosyu[:3])),
            help="*From Indeed Keyword Ranking.",
            )
        additionalKeyWordInputForm = st.text_input(
            label="追加で調べたいキーワードを入力してください",
            value="",
            max_chars=100,
            placeholder="e.g. keyword1,keyword2,keyword3",
           )
        defaultData = st.checkbox(label="既存データで計算",value=True,)
        st.session_state["isPressedSR"] = button_states()
        srConfirmButton = st.form_submit_button("キーワード確定")

    candidateKeyWord = keyWordSelectForm
    if additionalKeyWordInputForm != "":
        candidateKeyWord += additionalKeyWordInputForm.split(",")
    candidateKeyWord = list(set([e for e in candidateKeyWord if len(e) > 0]))

    ########## 関連度計算フェース
    if srConfirmButton and defaultData:

        st.session_state["isPressedSR"]["pressed"] = True
        txtTitleSR, txtContentSR = readUploadedFile()


    elif srConfirmButton and not defaultData:

        st.session_state["isPressedSR"]["pressed"] = True
        txtTitleSR, txtContentSR = readUploadedFile()


########## Ginza
def ginzaed(expression):

    payload = {"inputs" : expression,}

    response = client.invoke_endpoint(
        EndpointName = 'ginzaElectra',
        ContentType = 'application/json',
        Body = json.dumps(payload),
        )

    result = json.loads(response["Body"].read().decode())
    resultTxtList = [e["generated_text"] for e in result]

    return resultTxtList

@st.cache
def loadCorpus(corpath):
    dfKaigoSponsor = pd.read_csv(corpath)
    kaigoSponsorSentList = list(
        itertools.chain(
            *[re.split("[。\n]", article) for article in dfKaigoSponsor.jobDescriptionText.tolist()]
            ))
    kaigoSponsorSentSet = [e.strip() for e in set(kaigoSponsorSentList) if len(e) > 0]
    return kaigoSponsorSentSet

#################### Expression Recommandation    
if optionPhase == "原稿推薦表現":# and task_submitted:

    homepageHolder.empty()    
    expRecContainer = expRecHolder.container()
    expRecContainer.markdown(str_block_css,unsafe_allow_html=True)
    expRecContainer.markdown("""
        <h1 style="text-align:start;">
            原稿<font color="deepskyblue">推薦</font>表現
                <sub class="pagetitle">&nbsp;<font color="deepskyblue">R</font>ecommended <font color="deepskyblue">E</font>xpression
                </sub></h1>
        """,unsafe_allow_html=True)


    if optionGyosyu == "介護":
        pass
        #gyoSyuSents = loadCorpus(corpath)
    elif optionGyosyu == "物流":
        pass
        #gyoSyuSents = loadCorpus(corpath)

    testginza = ginzaed("今日はいい天気です")
    st.write(testginza)


########## GPT-2 rinna モデル
      
def jobPostingGenerator(expression):

    payload = {
        "inputs" : expression,
        "parameters" : {
            "top_k" : 100,
            "return_full_text" : False,
            "num_return_sequences" : 10,
            },
        }

    response = client.invoke_endpoint(
        EndpointName = 'rinnaJapaneseGPT2Medium',
        ContentType = 'application/json',
        Body = json.dumps(payload),
        )

    result = json.loads(response["Body"].read().decode())
    resultTxtList = [e["generated_text"] for e in result]

    # resultTxtList = []
    # for e in result:
        # resultTxtList += [e2 for e2 in e["generated_text"].split("。")[:-1] if len(e2) > 2]

    return resultTxtList


#################### JOB POSTING GENERATION    
if optionPhase == "原稿生成β":

    homepageHolder.empty()    
    jobGenContainer = jobGenHolder.container()
    jobGenContainer.markdown(str_block_css,unsafe_allow_html=True)
    jobGenContainer.markdown("""
        <h1 style="text-align:start;">
            原稿<font color="deepskyblue">生成</font>表現
                <sub class="pagetitle">&nbsp;<font color="deepskyblue">J</font>ob <font color="deepskyblue">P</font>osting <font color="deepskyblue">G</font>eneration
                </sub></h1>
        """,unsafe_allow_html=True)
    
    ########## sentence input form
    with jobGenContainer.form("jobgenform"):

        leader = st.text_area(
            label = "",
            max_chars = 200,
            help = "rinna GPT-2",
            placeholder = "表現に困っている要望・アピールポイント等を入力してください",
            )
        generatedLength = st.slider(
            label = "文字数を決めてください",
            min_value = 1,
            max_value = 256,
            value = (3,100),
        )
        
        jobGenSubmitted = st.form_submit_button("生成")

    if jobGenSubmitted:

        generatedTxtList = jobPostingGenerator(leader)
        for paragraph in generatedTxtList:
            st.markdown(f"""
                <div class="parblock">{paragraph}</div><p></p>
                """,unsafe_allow_html=True)



            
