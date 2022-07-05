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
import plotly.graph_objects as go
import plotly.express as px

#################### html wrapper & page config
session = Session(profile_name="genikko-profile")
client = session.client("sagemaker-runtime",region_name="ap-northeast-1")
clientLambda = session.client("lambda",region_name="ap-northeast-1")

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
        span.strblockGray{{
            padding         :   8px;
            /*border          :   1px solid lightskyblue;*/
            border-radius   :   10px;
            background      :   #f3f4f7;
            font-size       :   110%;
        }}
        span.strblockBlue{{
            padding         :   6px;
            border          :   1.5px solid lightskyblue;
            border-radius   :   10px;
            /*background      :   lightskyblue;*/
            font-size       :   80%;
            margin          :   5px;
            display         :   inline-block;
        }}
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
st.set_page_config(
    page_title="原稿解析 AIRCreW",
    page_icon="./materials/favicon.ico",
    layout="centered",
    initial_sidebar_state="auto",
    )
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def noSymbolic(sentence):
    
    replacements = [
        ('&lt;', ''), ('&gt;', ''), ('&amp;', ''),
        ('https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '[U]'),
        ('www?[\w/:%#\$&\?\(\)~\.=\+\-]+', '[U]'),
        ('\b\d{1,3}(,\d{3})*\b', '[N]'), ('\d+?(万円|円|万)', '[S]'), ('\d+?分', '[D]'), ('\d{1,2}:\d{2}', '[T]'), ('\d+', '0'),
        ('[◆♪●◎〜 ゚■✅️⭐≫≪※★┗━┃＃【】「」、。＝～☆＜＼：✿「」〔〕“”〈〉《》*『』【】＆＊・（）＄＃＠？！｀＋￥％�｜∴∵／°✧＞→｢｣₊⁎˳⁺༚①②◇｡･ﾟ…▼↓]',' '),
        ('[!-/:-@[-`{-~]','~SYM~'), ('(~SYM~)+?',' '),
        ]
    
    for old,new in replacements:
        sentence = re.sub(old, new, sentence)

    return " ".join(sentence.split()).strip()
    
def sidebarDisplay(taskLabel):

    st.sidebar.image("./materials/VQlogo.png")
    st.sidebar.image(image="./materials/AIRCreW_logo.PNG",width=300)

    if "industry" in st.session_state:
        gyosyuDisplay = st.session_state.industry
    else: gyosyuDisplay = "-"

    st.markdown(str_block_css,unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <p style="text-align: center">現在の業種：<span class="strblock"><font color="white">{gyosyuDisplay}</font></span></p>
        """, unsafe_allow_html=True,)

    st.sidebar.markdown(str_block_css,unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <p style="text-align: center">現在のタスク：<span class="strblock"><font color="white">{taskLabel}</font></span></p>
        """, unsafe_allow_html=True,)
    
    with st.sidebar.expander("このアプリについて"):
        st.write("""
            AI(人工知能)のチカラを使ってindeed原稿を解析・改善する独自ツールを展開しています。
            原稿の比較検証等を可視化することで、改善ポイントを明確にして最適なindeed原稿へと導くことが可能です。
        """)

    st.sidebar.markdown('<span style="font-size:12px;text-align: center"><b>Copyright ©2022 株式会社一広バリュークエスト</b></span>',unsafe_allow_html=True)

sidebarDisplay(taskLabel="-")

def readUploadedFile(fileUploaderForm):

    if fileUploaderForm is not None:
        txt = StringIO(
            fileUploaderForm.getvalue().decode("utf-8")
            ).read()
    else:
        st.warning("No file detected. Will use the default data for demonstration purposes.")
        with open("./testIkko.txt",encoding="utf-8") as fr:
            txt = fr.read()
    
    st.session_state.target_file = txt
    st.session_state.url = "default file"
    
    return txt

def titleContent(txt):

    title = txt.split("\n")[0]

    content = []
    for e in txt.split("\n")[1:]:
        eNoSym = noSymbolic(e).strip()
        if len(eNoSym) > 0:
            content.append(eNoSym)
    content = "\n".join(content)

    return title, content

########## H3 ファイルをアップロード

st.markdown('<h3 style="text-align: center">ファイルをアップロード</h4>',unsafe_allow_html=True)

if "target_file" not in st.session_state:
    st.session_state.target_file = 0

with st.form("uploaderSingleFile", clear_on_submit=True,):

    ## 1. from uploaded files
    fileUploaderForm = st.file_uploader(label="")
    submitted = st.form_submit_button(label="アップロード")

if fileUploaderForm and submitted:
    st.success("ファイルをアップロードしました。")
    if "url" not in st.session_state:
        st.session_state.url = ""
    st.session_state.url = "upload file"

readUploadedFile(fileUploaderForm)

########## H3 業種選択
generalKeywords = [
    "正社員","アルバイト","中高年","主婦パート","土日祝休み",
    "シニア","ハローワーク","深夜バイト","オープニングスタッフ",
    "未経験","主婦","パート","短期","大学生","高校生","早朝",
    "単発","託児所完備",
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
        "販売","アパレル販売","販売スタッフ","ファッション","デザイン",
        "コラボ","メンズ","レディース","デザイナー","接客","接客販売",
        "カフェ","パン屋","品出し","スーパー","ドラッグストア","喫茶店",
        "バリスタ","コーヒー","雑貨販売",
        ],
    "営業":[
        "営業","不動産",
        ],
    "飲食":[
        "飲食","調理補助","キッチンスタッフ","ホール","調理","給食","食堂",
        "洗い場","飲食店","居酒屋","ベーカリー","パン製造","食品製造",
        ],
    "事務":[
        "事務",
        ],
    "マーケティング":[
        "ecサイト","運営スタッフ","企画営業","広報","企画","商品企画",
        "ネットショップ","運営","在宅勤務","在宅ワーク","通販",
        "photoshop","秘書","ラウンダー","商品開発","ものづくり",
        ]
    }

st.markdown('<h3 style="text-align: center">業種を選択してください</h4>',unsafe_allow_html=True)
optionGyosyu = st.selectbox(label="", options=["-"] + list(gyosyuKeywordDict.keys()))
gyosyuStore = optionGyosyu

if "industry" not in st.session_state:
    st.session_state.industry = "notSelected"
elif st.session_state != optionGyosyu:
    st.session_state.industry = gyosyuStore

def ginzaProcessing(task="singleText",sent1="",sent2=""):

    payload = {
        "task" : task,
        "singleText" : sent1,
        "pairText" : [
            sent1,
            sent2
            ]
        }

    response = clientLambda.invoke(
        FunctionName = 'arn:aws:lambda:ap-northeast-1:836835325374:function:test',
        InvocationType = 'RequestResponse',
        #InvocationType = 'Event',
        Payload = json.dumps(payload),
        )

    result = json.loads(response["Payload"].read().decode("utf-8"))

    return result

def multiAddDiv(df):
    result1 = sum(df["原稿文数"]*df["文平均字数"])/sum(df["原稿文数"])
    result2 = sum(df["原稿文数"]*df["文平均語数"])/sum(df["原稿文数"])
    result3 = sum(df["原稿文数"]*df["文平均名詞数"])/sum(df["原稿文数"])
    return [result1,result2,result3]

def pending_docAveRec(df):# なぜかコラムずれる？

    dfStatMean = df.mean().tolist()[1:-3]
    dfStatMean4Rec = ["dummy0","dummy1"] + dfStatMean + multiAddDiv(df)
    st.session_state.statBackgroundAvg = dfStatMean4Rec

    return dfStatMean4Rec

def forSentence(s):

    s = s.strip()
    sParsed = ginzaProcessing(task="singleText",sent1=s)
    sWakati = [e[0] for e in sParsed["info"] if not e[1].startswith("補助記号")]
    sNoun = [e[0] for e in sParsed["info"] if e[1].startswith("名詞")]

    return [len(s),len(sWakati),len(sNoun)]

def forDescription(para):

    para = para.strip()
    sentlist = [sent for sent in para.split("\n") if len(sent) > 0]
    paraParsed = ginzaProcessing(task="singleText",sent1=para)
    paraWakati = [e[0] for e in paraParsed["info"] if not e[1].startswith("補助記号")]
    paraNoun = [e[0] for e in paraParsed["info"] if e[1].startswith("名詞")]
    resultsPara = [len(para),len(paraWakati),len(set(paraWakati)),len(paraNoun),len(set(paraNoun)),len(sentlist),]
    resultsSent = []

    for sent in sentlist:
        resultsSent.append(forSentence(sent))
    sMean = np.mean(resultsSent, axis=0)

    results = resultsPara + sMean.tolist()
    
    return results

def getDeviationValue(df,colName):

    seriesCal = df[colName]
    seriesCal_std = seriesCal.std(ddof=0)
    seriesCal_mean = seriesCal.mean()
    result = seriesCal.map(lambda x: round((x - seriesCal_mean) / seriesCal_std * 10 +50)).astype(int).tolist()

    return result

def statTargetDoc(_txtTitle,_txtContent):

    statTitle = forSentence(_txtTitle)
    statContent = forDescription(_txtContent)
    record = ["dummy0","dummy1",*statTitle,*statContent]

    st.session_state.statTargetTxt = record

    return record

def radar_chart(dataRadarChart,categoryRadarChart):       
    def closeline(i):
        rlist = dataRadarChart.iloc[i].tolist()
        rlist.append(rlist[0])
        rr = rlist
        return rr
    
    categoryRadarChart.append(categoryRadarChart[0])
    categories = categoryRadarChart
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=closeline(-1),
        theta=categories,
        line=dict(color="black",width=3),
        name="対象原稿", 
        fill="toself",
        ))
    
    fig.add_trace(go.Scatterpolar(
        r=closeline(-2),
        theta=categories,
        line=dict(color="deepskyblue",width=2,dash="dot"),
        name="有料原稿", 
        fill="toself",
        ))

    fig.update_layout(
        margin=dict(l=64,r=64,b=10,t=10),
        #paper_bgcolor="lightskyblue",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.00,1.00],
            )),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.8,
            xanchor="left",
            x=0.01,
        ),
        showlegend=True,
        )
    
    return fig

def getSimValue(dic4store,kwlist,targetTitle,targetDoc):
    dic4store["職種"].append()
    for kw in kwlist:
        simScore = ginzaProcessing(
            task="pairText",
            sent1=kw,
            sent2=targetDoc)["cosine_similarity"]
        dic4store[kw].append(simScore)
    return dic4store

def loadCorpusHP(corpath):
    dfKaigoSponsor = pd.read_csv(corpath)
    sentList = []
    for content in dfKaigoSponsor.jobDescriptionText.tolist():
        contentSplit = content.split(" ")
        sentList += contentSplit
    kaigoSponsorSentList = list(
        itertools.chain(
            *[article.split(" ") for article in dfKaigoSponsor.jobDescriptionText.tolist()]
            ))
    kaigoSponsorSentSet = [noSymbolic(e.strip()) for e in set(kaigoSponsorSentList) if len(e) > 0]
    kaigoSponsorSentSet = list(set(kaigoSponsorSentSet))
    return list(range(1000))
