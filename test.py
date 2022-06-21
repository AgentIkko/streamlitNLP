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

    def aboutUsContent():
        c1,c2,c3=st.columns([1,6,1])
        with c2: st.image("./indeed_logo_blue.png")
        st.markdown("<h1 style='text-align: center'>原稿解析・作成支援ツール</h1>",unsafe_allow_html=True)
        c1,c2,c3=st.columns([1,6,1])
        with c2: st.image("./AIRCreW_logo.PNG")
        st.markdown("<h2 style='text-align: center'><b>via</b></h2>",unsafe_allow_html=True)
        c1,c2,c3=st.columns([1,6,1])
        with c2: st.image("./VQlogo.png")

    with homepageHolder.container():
        aboutUsContent()

if infoPages == "HOME":
    homepageHolder.empty()
    ########### login module

#################### side bar
st.sidebar.markdown('<h4 style="text-align: center">ファイルをアップロード</h4>',unsafe_allow_html=True)

########## file upload
with st.sidebar.form("uploaderSingleFile", clear_on_submit=False,):

    ## 1. from uploaded files
    fileUploaderForm = st.file_uploader(label="① 原稿テキストファイル")
    submitted = st.form_submit_button("アップロード")

    if fileUploaderForm and submitted is not None:
        st.write("ファイルをアップロードしました。")


########## 業種選択
optionGyosyu = st.sidebar.selectbox(label="STEP 1: 業種選択", options=("-","介護","物流","販売","飲食","事務"))
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

#################### FSセクション用関数 ####################

### task : singleText | pairText
### singleText: str
### pairtText:[str1, str2]
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

@st.cache
def readUploadedFile():

    if fileUploaderForm is not None:
        #readUploadedFile.clear()
        txt = StringIO(fileUploaderForm.getvalue().decode("utf-8")).read()
    else:
        with open("./testIkko.txt",encoding="utf-8") as fr:
            txt = fr.read()
            #txt = """一広バリュークエストの独自サービス\n
            #AIRCreWで原稿解析。\n
            #一広バリュークエストに運用をまかせると求人費用30%以上削減できる。"""
    
    genkouTitle = txt.split("\n")[0]
    genkouContent = "\n".join([e.strip() for e in txt.split("\n")[1:] if len(e.strip())>0])

    return genkouTitle, genkouContent

@st.experimental_memo
def docAveRec(df):
    dfStatMean = df.mean().tolist()[1:-3]
    dfStatMean4Rec = ["dummy0","dummy1"] + dfStatMean + multiAddDiv(df)
    return dfStatMean4Rec

def forSentence(s):

    s = s.strip()
    sParsed = ginzaProcessing(task="singleText",sent1=s)
    sWakati = [e[0] for e in sParsed["info"] if not e[1].startswith("補助記号")]
    sNoun = [e[0] for e in sParsed["info"] if e[1].startswith("名詞")]

    return [len(s),len(sWakati),len(sNoun)]

def forDescrption(para):

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
                </sub></h1><hr>
        """,unsafe_allow_html=True)

    ########## 計算パート
    txtTitle, txtContent = readUploadedFile()

    #@st.experimental_singleton
    def statTargetDoc(_txtTitle,_txtContent):
        statTitle = forSentence(_txtTitle)
        statContent = forDescrption(_txtContent)
        return ["dummy0","dummy1",*statTitle,*statContent]
    
    targetDocRec = statTargetDoc(txtTitle,txtContent)
    dfBackground = pd.read_csv(f"./{optionGyosyu}_sponsorPro_stat.csv")
    lastRowRec = docAveRec(dfBackground)

    with funStaContainer:

        indexRange1 = [2,3,4,-3,-2,-1]
        indexRange2 = [5,6,7,8,9,10]
        labelRange = ["dummy1","dummy2","職種字数","職種語数","職種名詞数","原稿字数","原稿語数","原稿語数(異)","原稿名詞数","原稿名詞数(異)","原稿文数","文平均字数","文平均語数","文平均名詞数"]
        
        st.markdown(str_block_css,unsafe_allow_html=True)
        st.markdown(f"""
            <p>対象原稿職種：<span class="strblockGray">{txtTitle}</span></p>
            """, unsafe_allow_html=True,)
        for (i,col) in zip(indexRange1,st.columns(6)):
            targetNum = np.round(targetDocRec[i],decimals=1)
            deltaNum = np.round(targetDocRec[i]-lastRowRec[i],decimals=1)
            col.metric(labelRange[i],targetNum,deltaNum)

        st.markdown(str_block_css,unsafe_allow_html=True)
        st.markdown(f"""
            <p>対象原稿内容：<span class="strblockGray">{txtContent[:25]}（以下略）</span></p>
            """, unsafe_allow_html=True,)
        for (i,col) in zip(indexRange2,st.columns(6)):
            targetNum = np.round(targetDocRec[i],decimals=1)
            deltaNum = np.round(targetDocRec[i]-lastRowRec[i],decimals=1)
            col.metric(labelRange[i],targetNum,deltaNum)

    funStaContainer.markdown("<hr>", unsafe_allow_html=True)

    ########## グラフ出力パート
    @st.experimental_singleton
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

    lastRowRec_title = lastRowRec[2:5] + lastRowRec[-3:]
    lastRowRec_content = lastRowRec[5:-3]
    targetDocRec_title = targetDocRec[2:5] + targetDocRec[-3:]
    targetDocRec_content = targetDocRec[5:-3]

    # fig2 = px.histogram(df4RadarChart_title["タイトル字数"])
    # st.plotly_chart(fig2)
    col1,col2 = st.columns(2)

    with col1:#with funStaContainer:
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

    with col2:#with funStaContainer:
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
            "販売","アパレル販売","販売スタッフ",
            ],
        "営業":[
            "営業","不動産",
            ],
        "飲食":[
            "飲食",
            ],
        "事務":[
            "事務",
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

        srConfirmButton = st.form_submit_button("キーワード確定")

    candidateKeyWord = keyWordSelectForm
    if additionalKeyWordInputForm != "":
        candidateKeyWord += additionalKeyWordInputForm.split(",")
    candidateKeyWord = list(set([e for e in candidateKeyWord if len(e) > 0]))

    ########## 関連度計算フェース
    def getSimValue(dic4store,kwlist,targetDoc):
        for kw in kwlist:
            kw4SimCal = kw+"の求人"
            simScore = ginzaProcessing(
                task="pairText",
                sent1=kw4SimCal,
                sent2=targetDoc)["cosine_similarity"]
            dic4store[kw].append(simScore)
        return dic4store

    if srConfirmButton:

        txtTitleSR, txtContentSR = readUploadedFile()

        with st.spinner("データ処理中..."):
            ########## キーワード確定
            dictOfSimScores = {kw : [] for kw in candidateKeyWord}
            dictOfSimScores.update({"職種": []})

        try:
            simScoreData = pd.read_csv(f"phase2_{optionGyosyu}.csv") 
        except Exception:
            ########## 関連度計算
            dictOfSimScores["職種"].append("TARGET")
            dictOfSimScores = getSimValue(
                dic4store = dictOfSimScores,
                kwlist = candidateKeyWord,
                targetDoc = txtContentSR,)
            st.success("処理終了")

            dfSponsorProGenkou = pd.read_csv(f"{optionGyosyu}_sponsorPro_text.csv")
            contraTitles = dfSponsorProGenkou["jobTitle"].tolist()
            contraContents = dfSponsorProGenkou["jobDescriptionText"].tolist()

            #status_text = st.empty()
            #loadBarSR = st.progress(0)
            #loopCount = len(contraTitles)

            for (t,c) in zip(contraTitles,contraContents):
                dictOfSimScores["職種"].append(t)
                dictOfSimScores = getSimValue(
                    dic4store = dictOfSimScores,
                    kwlist =candidateKeyWord ,
                    targetDoc = c,)

            simScoreData = pd.DataFrame.from_dict(dictOfSimScores)
            simScoreData.to_csv(f"phase2_{optionGyosyu}.csv")

    with st.spinner("出力中..."):
        # lambda のほうで ent の出力を追加
        def getDeviationValue(df,colName):
            seriesCal = df[colName]
            seriesCal_std = seriesCal.std(ddof=0)
            seriesCal_mean = seriesCal.mean()
            result = seriesCal.map(lambda x: round((x - seriesCal_mean) / seriesCal_std * 10 +50)).astype(int).tolist()
            return result

        try:
            for kw in candidateKeyWord:
                sss = simScoreData[kw].rank(
                    ascending=True,
                    pct=True).tolist()[0]
                #sssDV = getDeviationValue(simScoreData,kw)[0]
                st.metric("偏差値",sss)
                
        except NameError:
            st.info("キーワードをまず選択してください。\n")     

@st.experimental_memo
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
                </sub></h1><hr>
        """,unsafe_allow_html=True)


    if optionGyosyu == "介護":
        pass
        #gyoSyuSents = loadCorpus(corpath)
    elif optionGyosyu == "物流":
        pass
        #gyoSyuSents = loadCorpus(corpath)

    testginza = ginzaProcessing(task="singleText",sent1="今日はいい天気です")
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



            
