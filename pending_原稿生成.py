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