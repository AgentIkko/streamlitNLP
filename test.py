from io import StringIO
from PIL import Image
from collections import Counter
import re, os, glob, copy, random, pickle, itertools
import ginza, spacy
import streamlit as st
import pandas as pd
import numpy as np

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
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
    page_title="原稿解析システム_DEMO",
    #page_icon=":smiley_cat:",
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
    
#################### sidebar
st.sidebar.image("./VQlogo.png")
st.sidebar.image(image="./AIRCreW_logo.PNG",width=300)

#################### homepage

homepageHolder = st.empty()
aboutusHolder = st.empty()
funStaHolder = st.empty()
semRelHolder = st.empty()
expRecHolder = st.empty()

infoPages = st.sidebar.radio("",options=(["HOME","ABOUT US"]))

if infoPages == "ABOUT US":
    funStaHolder.empty()
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
# STEP 1
# 1. indeed 仕事 uid 入力 jobUrlInputForm
# 2. 買ってきた json ファイル入力 | pending
# 3. indeed 一括入稿フォーマットでの入力 | pending
st.sidebar.markdown('<h4 style="text-align: center">STEP 1: アップロード方式を選択</h4>',unsafe_allow_html=True)
with st.sidebar.form("uploaderSingleFile",clear_on_submit=False):

    ## 1. from uploaded files

    fileUploaderForm = st.file_uploader(label="① 原稿テキストファイル")
    submitted = st.form_submit_button("アップロード")

    if fileUploaderForm and submitted is not None:
        st.write("ファイルをアップロードしました。")

    ## 2. from indeed urls
    
    jobUrlInputForm = st.text_input(label="② uidを入力",value="",max_chars=50,)
    url_submitted = st.form_submit_button("案件URL獲得")

    if jobUrlInputForm and url_submitted is not None:
        jobUrl = "https://jp.indeed.com/%E4%BB%95%E4%BA%8B?jk=" + jobUrlInputForm
        st.markdown("[indeed URL]("+jobUrl+")",unsafe_allow_html=True)    


# STEP 2
st.sidebar.markdown('<h4 style="text-align: center">STEP 2: 解析モデルと業種を選択</h4>',unsafe_allow_html=True)
with st.sidebar.form("modelGyosyu",clear_on_submit=False):
    # model select
    optionModel = st.selectbox(label="利用可能モデル", options=("W2V","RNN","GenIkko",))
    optionGyosyu = st.selectbox(label="業種", options=("介護","物流","販売"))
    mg_submitted = st.form_submit_button("確定")
    st.session_state["isPressedModel"] = button_states()

    if mg_submitted:
        st.session_state["isPressedModel"].update({"pressed":True})
        st.write("モデルをロードしました。")
    

# STEP 3
st.sidebar.markdown('<h4 style="text-align: center">STEP 3: 解析タスクを選択</h4>',unsafe_allow_html=True)
# with st.sidebar.form("taskSelect",clear_on_submit=False):
optionPhase = st.sidebar.selectbox(label="処理タスク",options=("-","基礎統計","関連度計算","原稿推薦表現"))
st.sidebar.write("現在のタスク：", optionPhase)
#task_submitted = st.form_submit_button("解析開始")
#    st.session_state["isPressedTask"] = button_states()
#    if task_submitted:
#        st.session_state["isPressedTask"]["pressed"] = True
   
with st.sidebar.expander("このアプリについて"):
    st.write("""
        AI(人工知能)のチカラを使ってindeed原稿を解析・改善する独自ツールを展開しています。
        原稿の比較検証等を可視化することで、改善ポイントを明確にして最適なindeed原稿へと導くことが可能です。
    """)
    
st.sidebar.markdown('<span style="font-size:12px;text-align: center"><b>Copyright ©2022 株式会社一広バリュークエスト</b></span>',unsafe_allow_html=True)

#################### functions-fs
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

#################### FS
if optionPhase == "基礎統計":# and task_submitted:

    homepageHolder.empty()
    funStaContainer = funStaHolder.container()
    funStaContainer.title("Fundamental Statistics")

    funStaContainer.write("HELLO")

#################### functions-sr

def getDeviationValue(df,colName):

    seriesCal = df[colName]
    seriesCal_std = seriesCal.std(ddof=0)
    seriesCal_mean = seriesCal.mean()
    result = seriesCal.map(lambda x: round((x - seriesCal_mean) / seriesCal_std * 10 +50)).astype(int).tolist()

    return result

#################### SR
if optionPhase == "関連度計算":
    
    homepageHolder.empty()
    st.title("Semantic Relevance")
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
            ]
        }

    for kw in gyosyuKeywordDict.keys():
        if optionGyosyu == kw:
            selectedGyosyu = gyosyuKeywordDict[kw]

    optionKeywords = generalKeywords + selectedGyosyu

    ########## 関連度を調べたいキーワードの選択と自由入力
    with st.form("keywordselect"):
        keyWordSelectForm = st.multiselect(
            label="キーワード",
            options=optionKeywords,
            default=generalKeywords[:5] + selectedGyosyu[:5],
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




#################### Expression Recommandation    
if optionPhase == "原稿推薦表現":# and task_submitted:
    homepageHolder.empty()    
    expRecContainer = expRecHolder.container()
    expRecContainer.title("Expression Recommendation")

    ########## load nlped sentences
    #@st.experimental_singleton
    #def pickle2List():
        #pped = []
        #pklPaths = glob.glob("./kaigoSponsorSentNLPed/*.pkl")
        #for p in pklPaths:
        #    pped += picklePick(p)
        #pped = picklePick("./kaigoSponsorSentNLPed/kaigoSponsorSentNLPed_5.pkl")
        #return pped
