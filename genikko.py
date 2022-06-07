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
    funStaContainer.markdown("<h2 style='text-align: start; color: royalblue;'>対象原稿の基礎統計量</h2><hr>", unsafe_allow_html=True)
    # funStaContainer.title("対象原稿の基礎統計量")

    dfSponsorStat = pd.read_csv("./data_pandas/kaigo_sponsor_stat.csv")
    dfSponsorMean = dfSponsorStat.mean().tolist()[1:]
    dfSponsorMeanRecord = ["x","x"] + dfSponsorMean[:-3] + multiAddDiv(dfSponsorStat)

    dfSponsorProStat = pd.read_csv("./data_pandas/kaigo_sponsorPro_stat.csv")
    dfSponsorProMean = dfSponsorProStat.mean().tolist()[1:]
    dfSponsorProMeanRecord = ["x","x"] + dfSponsorProMean[:-3] + multiAddDiv(dfSponsorProStat)
    
    import plotly.express as px
    import plotly.graph_objects as go

    #@st.cache
    def forSentence(s):
        s = s.strip()
        sNLP = nlp(s)
        sWakati = [token.text for token in sNLP if not token.tag_.startswith("補助記号")]
        sNoun = [token.text for token in sNLP if token.tag_.startswith("名詞")]
        return [len(s),len(sWakati),len(sNoun)]

    #@st.cache
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

    ########### 処理part-1
    txtTitle,txtContent = readUploadedFile()

    charTitle, wordTitle, nounTitle = forSentence(txtTitle)    
    charG, wordG, wordGdiff, nounG, nounGdiff, sentG, sentCharG, sentWordG, sentNounG = forDescrption(txtContent)
    targetRecord = ["x", "x", *forSentence(txtTitle), *forDescrption(txtContent)] 

    dfFS = dfSponsorStat
    for e in [dfSponsorMeanRecord,dfSponsorProMeanRecord,targetRecord]:
        dfFS = dfFS.append(e)
    
    with funStaContainer:
        st.write("解析対象原稿職種：",txtTitle)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("職種字数",charTitle,np.round(charTitle-dfSponsorProMean[0],decimals=1))
        col2.metric("職種語数",wordTitle,np.round(wordTitle-dfSponsorProMean[1],decimals=1))
        col3.metric("職種名詞数",nounTitle,np.round(nounTitle-dfSponsorProMean[2],decimals=1))
        col4.metric("文平均字数",np.round(sentCharG,decimals=1),np.round(sentCharG-dfSponsorProMean[-3],decimals=1))
        col5.metric("文平均語数",np.round(sentWordG,decimals=1),np.round(sentWordG-dfSponsorProMean[-2],decimals=1))
        col6.metric("文平均名詞数",np.round(sentNounG,decimals=1),np.round(sentNounG-dfSponsorProMean[-1],decimals=1))
        
        st.write("解析対象原稿内容：",txtContent[:31],"（以下略）")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("原稿字数",charG,np.round(charG-dfSponsorProMean[3],decimals=1))
        col2.metric("原稿語数",wordG,np.round(wordG-dfSponsorProMean[4],decimals=1))
        col3.metric("原稿語数(異)",wordGdiff,np.round(wordGdiff-dfSponsorProMean[5],decimals=1))
        col4.metric("原稿名詞数",nounG,np.round(nounG-dfSponsorProMean[6],decimals=1))
        col5.metric("原稿名詞数(異)",nounGdiff,np.round(nounGdiff-dfSponsorProMean[7],decimals=1))
        col6.metric("原稿文数",sentG,np.round(sentG-dfSponsorProMean[8],decimals=1))

    funStaContainer.markdown("<hr>", unsafe_allow_html=True)

    ########### 処理part-2    
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
            r=closeline(2),
            theta=categories,
            line=dict(color="black",width=3),
            name="対象原稿", 
            ))
        
        fig.add_trace(go.Scatterpolar(
            r=closeline(0),
            theta=categories,
            line=dict(color="steelblue",width=2,dash="dot"),
            name="有料原稿", 
            ))
        
        fig.add_trace(go.Scatterpolar(
            r=closeline(1),
            theta=categories,
            line=dict(color="darkorange",width=2,dash="dot"),
            name="優良原稿", 
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.00,1.00],
                )),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="left",
                x=0.23,
            ),
            showlegend=True,
            )
        
        return fig
       
    # title part
    with funStaContainer:
        st.markdown("<h3 style='text-align: center; color: black;'>文単位解析</h3>", unsafe_allow_html=True)
        
        # colTitleRadarChart,colTitleTable = st.columns([5,2])
        # with colTitleRadarChart:
            
        dfTitleRC = dfFS[["タイトル字数","タイトル語数","タイトル名詞数","文平均字数","文平均語数","文平均名詞数"]].rank(pct=True)
        figRadarTitle = radar_chart(dfTitleRC,categoryRadarChart = ["職種字数","職種語数","職種名詞数","平均字数","平均語数","平均名詞数"])
        st.plotly_chart(figRadarTitle,use_container_width=True)
        
        with st.expander("データ"):
            st.dataframe(
                data = dfSponsorProStat[["タイトル字数","タイトル語数","タイトル名詞数","文平均字数","文平均語数","文平均名詞数"]],#.assign(hack='').set_index('hack'),
                height=400
                )
    
    # body part
    with funStaContainer:
        st.markdown("<h3 style='text-align: center; color: black;'>原稿単位解析</h3>", unsafe_allow_html=True)
        
        # colBodyRadarChart,colBodyTable = st.columns(2)
        # with colBodyRadarChart:
        dfBodyRC = dfFS[["原稿字数","原稿語数","原稿異なり語数","原稿名詞数","原稿異なり名詞数","原稿文数"]].rank(pct=True)
        figRadarBody = radar_chart(dfBodyRC,categoryRadarChart = ["原稿字数","原稿語数","原稿異なり語数","原稿名詞数","原稿異なり名詞数","原稿文数"])
        st.plotly_chart(figRadarBody,use_container_width=True)

            
        with st.expander("データ"): 
            st.dataframe(
                data = dfSponsorProStat[["原稿字数","原稿語数","原稿異なり語数","原稿名詞数","原稿異なり名詞数","原稿文数"]].astype(int),
                height=400
                )


#################### functions-sr

def simScore4singleGenkou(dic,kwlist,gkTitle,gkContent):

    dic["職種"].append(gkTitle)
    nlped = nlp(gkContent)
    for kw in kwlist:
        kw4sim = kw+"の求人"
        simScore = nlp(kw4sim).similarity(nlped)
        dic[kw].append(simScore)

def getDeviationValue(df,colName):

    seriesCal = df[colName]
    seriesCal_std = seriesCal.std(ddof=0)
    seriesCal_mean = seriesCal.mean()
    result = seriesCal.map(lambda x: round((x - seriesCal_mean) / seriesCal_std * 10 +50)).astype(int).tolist()

    return result

def semRelExpanderContent(kw,dv,sentlist):
    with st.expander(kw):

        nlpedKW = nlp(kw)
        simScores = []

        with st.spinner("計算中..."):
            for s in sentlist:
                sNLP = nlp(s)
                sNew = "".join([token.text for token in sNLP if not token.tag_.startswith("補助記号")])
                if len(sNew) > 0:
                    simScore = nlpedKW.similarity(nlp(sNew))
                    simScores.append((sNew,simScore))
            sortedSimScores = sorted(simScores,key=lambda x: x[1], reverse=True)
            sortedSimSocres = [e for e in sortedSimScores if len(e[0]) >= 5]

        col1,col2 = st.columns([1,3])

        with col1:
            st.metric("偏差値", dv)
        
        with col2:
            st.markdown(f"""
                <div style = "border-radius:4px 4px 4px 4px;text-align:start;background-color:#e0f0d8;"> {sortedSimScores[0][0]}</div>
                <p style = "margin-bottom: 0.5px"><p>
                <div style = "border-radius:4px 4px 4px 4px;text-align:start;background-color: #e0f0d8"> {sortedSimScores[1][0]}</div>
                <p style = "margin-bottom: 0.5px"><p>
                <div style = "border-radius:4px 4px 4px 4px;text-align:start;background-color: #e0f0d8"> {sortedSimScores[2][0]}</div>
                <p style = "margin-bottom: 1px"><p>
                <div style = "border-radius:4px 4px 4px 4px;text-align:start;background-color: #faeaea"> {sortedSimScores[-3][0]}</div>
                <p style = "margin-bottom: 0.5px"><p>
                <div style = "border-radius:4px 4px 4px 4px;text-align:start;background-color: #faeaea"> {sortedSimScores[-2][0]}</div>
                <p style = "margin-bottom: 0.5px"><p>
                <div style = "border-radius:4px 4px 4px 4px;text-align:start;background-color: #faeaea"> {sortedSimScores[-1][0]}</div>
                """,unsafe_allow_html=True)

def picklePick(fpath):
    with open(fpath,"rb") as fr:
        pickledfile = pickle.load(fr)
    return pickledfile

#################### SR
if optionPhase == "関連度計算":
    
    homepageHolder.empty()
    st.markdown("<h2 style='text-align: start; color: royalblue;'>キーワードとの関連度</h2>", unsafe_allow_html=True)
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
        existedRelData = pd.read_csv("./data_pandas/kaigo_sponsor_Rel.csv")
        txtTitleSR, txtContentSR = readUploadedFile()


    elif srConfirmButton and not defaultData:

        st.session_state["isPressedSR"]["pressed"] = True
        txtTitleSR, txtContentSR = readUploadedFile()

        with st.spinner("処理中..."):
            ########## キーワード確定
            dictOfSimScores = {kw : [] for kw in candidateKeyWord}
            dictOfSimScores.update({"職種":[]})

            ########## 関連度計算
            simScore4singleGenkou(dictOfSimScores,candidateKeyWord,"TARGET",txtContentSR)
            st.success("対象原稿計算終了")

            dfSponsorProGenkou = pd.read_csv("./data_pandas/kaigo_sponsorPro_text.csv")
            contraTitles = dfSponsorProGenkou["jobTitle"].tolist()
            contraContents = dfSponsorProGenkou["jobDescriptionText"].tolist()

            status_text = st.empty()
            loadBarSR = st.progress(0)
            loopCount = len(contraTitles)

            for i,(t,c) in enumerate(zip(contraTitles,contraContents)):
                simScore4singleGenkou(dictOfSimScores,candidateKeyWord,t,c)
                status_text.text(f"職種：{t[:25]}...\t Progress: {round(i/loopCount*100,2)}%")
                loadBarSR.progress(int(i/loopCount*100+1))

            existedRelData = pd.DataFrame.from_dict(dictOfSimScores)
            st.success("DONE")
 
    ########## 出力フォーマット
    with st.spinner("解析完了。出力中..."):

        # st.write("めちゃめちゃ正規分布してるので、偏差値出してみました。")

        import plotly.figure_factory as ff
        from matplotlib import pyplot as plt

        genkouSentList = [e.strip() for e in txtContentSR.split("\n") if len(e) > 0]

        try:
            for kw in candidateKeyWord:
                sss = existedRelData[kw].rank(pct=True).tolist()[0]
                sssDV = getDeviationValue(existedRelData,kw)[0]
                semRelExpanderContent(kw,sssDV,genkouSentList)
                #st.write(kw,": 順位 ",round(sss*100,2),"%; 偏差値 ",sssDV)
                #fig = plt.figure()
                #ax = fig.add_subplot()
                #ax.hist(existedRelData[kw],bins=100)
                #st.pyplot(fig)
        except NameError:
            st.info("""
                まずはキーワードを選択してください。\n
                """)

        #st.write(dictOfSimScores)


#################### Expression Recommandation    
if optionPhase == "原稿推薦表現":# and task_submitted:
    homepageHolder.empty()    
    expRecContainer = expRecHolder.container()
    expRecContainer.markdown("<h2 style='text-align: start; color: royalblue;'>原稿推薦表現</h2>", unsafe_allow_html=True)
    #expRecContainer.title("Expression Recommendation")

    ########## load nlped sentences
    #@st.experimental_singleton
    #def pickle2List():
        #pped = []
        #pklPaths = glob.glob("./kaigoSponsorSentNLPed/*.pkl")
        #for p in pklPaths:
        #    pped += picklePick(p)
        #pped = picklePick("./kaigoSponsorSentNLPed/kaigoSponsorSentNLPed_5.pkl")
        #return pped
    
    ########### load corpus
    @st.cache
    def loadCorpus():
        dfKaigoSponsor = pd.read_csv("./data_pandas/kaigo_sponsorPro_text.csv")
        kaigoSponsorSentList = list(
            itertools.chain(
                *[re.split("[。\n]", article) for article in dfKaigoSponsor.jobDescriptionText.tolist()]
                ))
        kaigoSponsorSentSet = [e.strip() for e in set(kaigoSponsorSentList) if len(e) > 0]
        return kaigoSponsorSentSet

    kaigoSents = loadCorpus()

    def recExpGet(sentlist,nlpedTarget):
        voidlist = []
        for nlpedpp in sentlist:
            simscore = nlpedTarget.similarity(nlp(nlpedpp))
            voidlist.append((nlpedpp,simscore))
        listPresent = sorted(voidlist,key=lambda x:x[1],reverse=True)#reverse=True)
        listPresent = [e[0] for e in listPresent if 4 < len(e[0]) < 26][:21]
        return listPresent

    #with st.spinner("Please waiting..."):
        #pp = pickle2List()
        #st.write(len(pp))
    #st.success("Completed.")

    with expRecContainer.form("expRecInput"):
        st.info("""
            以下の入力に対する推薦表現が得られる。
            """)
        # keywords
        expRecKeywordInputForm = st.text_input(
            label = "① KEYWORD(S)",
            max_chars = 50,
            placeholder = "e.g. keyword1,keyword2,keyword3,"
            )
        # 文
        expRecSentenceInputForm = st.text_area(
            label = "② SENTENCE(S)",
            height = 10,
            max_chars = 200,
            placeholder = "文末判定：「。」、改行",
            )
        # uploaded doc
        expRecArtiInputForm = st.radio(
            label = "③ アップロード原稿の利用",
            options = ("利用する","利用しない"),
            index = 1,
            horizontal = True,
            )
        if expRecArtiInputForm == "利用する":    
            expRecTitleUploaded, expRecContentUploaded = readUploadedFile()

        expRecTargetConfirmButton = st.form_submit_button("対象確定")

    if expRecTargetConfirmButton:

        expRecKwList = [e+"の求人" for e in re.split("[,|]",expRecKeywordInputForm) if len(e) > 0]
        expRecSentList = [e for e in re.split('[。|\n]',expRecSentenceInputForm) if len(e) > 0]
        targetExpression = expRecKwList + expRecSentList

        try:
            expRecArti = expRecContentUploaded
            targetExpression.append(expRecArti)
            st.info("原稿利用あり")
        except NameError:
            st.info("原稿利用なし")

        if targetExpression == ["主婦","介護スタッフ"]:

            st.write("loading")

            import pickle
            with open("kaigoSyufuCalRes.p","rb") as fr:
                listSyufu = pickle.load(fr)
            with open("kaigoStaffCalRes.p","rb") as fr:
                listKaigoStaff = pickle.load(fr)

            #for candidate in ["主婦","介護スタッフ"]:
            with st.expander(label="主婦"):
                st.write(listSyufu[:10])
                with st.spinner("Processing..."):
                    listPresent = sorted(listSyufu,key=lambda x:x[1],reverse=True)
                    listPresent = [e[0] for e in listPresent if 4 <= len(e[0]) < 26][:21]
                    rawhtml = '<p style = "margin-bottom: 0.5px"></p>'.join(listPresent)
                    st.markdown(f"""
                        <div style = "border-radius:4px 4px 4px 4px;text-align:start;background-color:#e0f0d8;">{rawhtml}</div>
                        """, unsafe_allow_html=True)


        else:    
            for candidate in targetExpression:
                nlpedCandidate = nlp(candidate)
                with st.expander(label=candidate.replace('の求人','')):
                    with st.spinner("Processing..."):
                        voidlist = []
                        for nlpedpp in kaigoSents[:3000]:
                            simscore = nlpedCandidate.similarity(nlp(nlpedpp))
                            voidlist.append((nlpedpp,simscore))
                        listPresent = sorted(voidlist,key=lambda x:x[1],reverse=True)
                        listPresent = [e[0] for e in listPresent if 4 < len(e[0]) < 26][:21]
                        for s in listPresent:
                            st.write(s)
    
    else:
        st.info("""
            ロードが終わるまでお待ちください。""")

    
    
#################### Name Entity Recognition
#if optionPhase == "固有表現抽出":# and task_submitted:
#    homepageHolder.empty()    
#    st.title("Name Entity Recognition")
#    st.write("現在", optionModel,"_",optionGyosyu, "利用中")
#
#    with st.spinner("少々お待ちください"):
#        
#        nlp = spacy.load("ja_ginza")
#
#        dataStr = StringIO(fileUploaderForm.getvalue().decode("utf-8")).read()
#        dataNLPed = spacy.displacy.render(nlp(dataStr),style="ent")
#        st.write(HTML_WRAPPER.format(dataNLPed), unsafe_allow_html=True)
