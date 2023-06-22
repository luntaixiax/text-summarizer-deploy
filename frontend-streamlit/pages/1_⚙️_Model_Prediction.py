import os
from typing import List
import pandas as pd
import streamlit as st
from io import StringIO
from api import HandlerLocalBackend, HandlerMlflowBackend, HandlerLambdaBackend, HandlerSageMakerBackend, \
    ModelException, NotSupportedFunctionError, read_batch_pred_articles, write_batch_pred, \
    get_upload_pair, get_sample_article, get_sample_pair, log_summ, log_summs

def do_stuff_on_page_load():
    st.set_page_config(layout='centered')

do_stuff_on_page_load()

with st.sidebar:
    MODEL_CORE = st.radio('Select Model Core', options = ['Local', 'Mlflow', 'Lambda', 'SageMaker'], index=0, horizontal= False, 
        help='Local = local model artifact + fastapi; Mlflow = mlflow model server + fastapi'
    )
    if MODEL_CORE == 'Local':
        ip = st.text_input("Model IP", value = st.secrets['services']['local_service_name'], placeholder='`host` if standalone, otherwise service name')
        ip = '172.17.0.1' if ip == 'host' else ip
        port = st.number_input("Model Port", value = 8000)
        
        NUM_BEAN = st.slider('Num of Beans', min_value=5, max_value=10, value=8, step=1, disabled=(MODEL_CORE != 'Local'))
        TEMPERATURE = st.select_slider('Temperature', options=['low', 'medium', 'high'], value='medium', disabled=(MODEL_CORE != 'Local'))
        TEMPERATURE = dict(low=0.8, medium=1.0, high=1.2).get(TEMPERATURE)
        MODEL = HandlerLocalBackend(endpoint=f"http://{ip}:{port}", num_beans=NUM_BEAN, temperature=TEMPERATURE)
    elif MODEL_CORE == 'Mlflow':
        ip = st.text_input("Mlflow Backend IP", value = st.secrets['services']['mlflow_service_name'], placeholder='`host` if standalone, otherwise service name')
        ip = '172.17.0.1' if ip == 'host' else ip
        port = st.number_input("Mlflow Backend Port", value = 5000)
        NUM_BEAN = None
        TEMPERATURE = None
        MODEL = HandlerMlflowBackend(endpoint=f"http://{ip}:{port}")
    elif MODEL_CORE == 'Lambda':
        ip = st.text_input("Model URI", value = st.secrets['services']["lambda_endpoint"], 
                           placeholder='URI from API Gateway, starts with https')
        
        NUM_BEAN = st.slider('Num of Beans', min_value=5, max_value=10, value=8, step=1, disabled=(MODEL_CORE != 'Lambda'))
        TEMPERATURE = st.select_slider('Temperature', options=['low', 'medium', 'high'], value='medium', disabled=(MODEL_CORE != 'Lambda'))
        TEMPERATURE = dict(low=0.8, medium=1.0, high=1.2).get(TEMPERATURE)
        MODEL = HandlerLambdaBackend(endpoint=ip, num_beans=NUM_BEAN, temperature=TEMPERATURE)
    elif MODEL_CORE == 'SageMaker':
        endpoint_name=st.secrets['sagemaker']['sm_endpoint_name']
        st.text_input("Endpoint Name", value=endpoint_name, disabled=True)
        MODEL = HandlerSageMakerBackend(endpoint_name = endpoint_name)


def online_pred(text: str) -> str:
    try:
        summ_ = MODEL.summarize(article=text)
    except ModelException as e:
        st.session_state['online_pred_err'] = f"Model Exception: {e}"
    except NotSupportedFunctionError as e:
        st.session_state['online_pred_err'] = f"Not Supported Exception: {e}"
    except Exception as e:
        st.session_state['online_pred_err'] = e
    else:
        if 'online_pred_err' in st.session_state:
            del st.session_state['online_pred_err']
        st.session_state['online_pred_summ'] = summ_
    finally:
        st.session_state['online_pred_summ_expansion'] = True

def clear_online_pred():
    if 'online_pred_summ' in st.session_state:
        del st.session_state['online_pred_summ']
    st.session_state['online_pred_summ_expansion'] = False

def online_load_sample():
    article, summ = get_sample_article()
    st.session_state['online_inp'] = article
    st.session_state['online_target'] = summ
    clear_online_pred() # need to clear summary


def batch_pred(articles: List[str]) -> List[str]:
    try:
        summs = MODEL.summarize_batch(articles=articles)
    except ModelException as e:
        st.session_state['batch_pred_err'] = f"Model Exception: {e}"
    except NotSupportedFunctionError as e:
        st.session_state['batch_pred_err'] = f"Not Supported Exception: {e}"
    except Exception as e:
        st.session_state['batch_pred_err'] = e
    else:
        if 'batch_pred_err' in st.session_state:
            del st.session_state['batch_pred_err']
        st.session_state['batch_pred_summ'] = summs

def clear_batch_pred():
    if 'batch_pred_summ' in st.session_state:
        del st.session_state['batch_pred_summ']

def score_load_sample():
    #st.session_state['uploaded_pairs'] = get_sample_pair()
    sample_pair = get_sample_pair(num_sample = 5)
    st.session_state['upload_articles'] = sample_pair['article'].tolist()
    st.session_state['upload_targets'] = sample_pair['summ'].tolist()
    clear_batch_pred()


def score(pair: pd.DataFrame):
    try:
        scores = MODEL.scores(
            articles=pair['article'].tolist(), 
            targets=pair['summ'].tolist()
        ) # dict of [str, float]
    except ModelException as e:
        st.session_state['score_err'] = "😨 Model Exception: "
    except NotSupportedFunctionError as e:
        st.session_state['score_err'] = "🤡 Not Supported Exception: "
    except Exception as e:
        st.session_state['score_err'] = e
    else:
        if 'score_err' in st.session_state:
            del st.session_state['score_err']
        st.session_state['scores'] = scores

def online_log(article: str, summ: str, target: str, model_source:str, send_arize: bool = False):
    log_summ(
        article = article,
        summ = summ,
        target = target,
        model_source = model_source,
        send_arize = send_arize
    )
    st.balloons()

def batch_log(articles: List[str], summs: List[str], targets: List[str], model_source:str, send_arize: bool = False):
    log_summs(
        articles = articles,
        summs = summs,
        targets = targets,
        model_source = model_source,
        send_arize = send_arize
    )
    st.balloons()


if 'online_pred_summ_expansion' not in st.session_state:
    st.session_state['online_pred_summ_expansion'] = False # unexpand at the beginning
if 'online_pred_summ' not in st.session_state:
    st.session_state['online_pred_summ'] = "" # unexpand at the beginning
# if 'uploaded_pairs' not in st.session_state:
#     st.session_state['uploaded_pairs'] = pd.DataFrame()


tabs = st.tabs(['Online Prediction', 'Batch Prediction'])

with tabs[0]:
    text = st.text_area('Input your News here', value="", height=250, 
            max_chars=512 * 5, key='online_inp') # clear when changed
    btn_cols = st.columns([2, 9])
    with btn_cols[0]:
        online_load_sample_btn = st.button('Load Sample', on_click=online_load_sample, type="secondary",
                                    key = "online_load_sample_btn")
    with btn_cols[1]:
        online_pred_btn = st.button('Summarize', on_click=online_pred, kwargs=dict(text=text), type="primary",
                                    key = "online_pred_btn")
        
    with st.expander("Summarization", expanded = st.session_state['online_pred_summ_expansion']):
        if 'online_pred_err' in st.session_state:
            st.error(st.session_state['online_pred_err'])
            summ = ""
        else:
            summ = st.text_area(label = "✏️✏️✏️✏️✏️", placeholder="Click Summarize to See the Results", 
                            value = st.session_state['online_pred_summ'], disabled=True)
            
    target_online = st.text_area('Input Ground Truth here (if you know)', value="", 
                                 key='online_target')
    if target_online != "" and summ != "":
        online_send_to_arize = st.checkbox(
            label = "Send to Arize?",
            value = False,
            key = "online_send_to_arize"
        )
        online_log_btn = st.button(
            'Log Summarization', 
            on_click=online_log,
            key = "online_log_btn",
            kwargs=dict(
                article=text, 
                summ = summ, 
                target = target_online, 
                model_source = MODEL_CORE,
                send_arize = online_send_to_arize
            ),
            type="secondary")

            
with tabs[1]:
    st.info(body = "You can load sample pairs. Or you can choose to upload your own CSV pairs", icon = "📰")
    score_sample_cols = st.columns(2)
    with score_sample_cols[0]:
        upload_file = st.file_uploader("Upload articles", type=['csv'], 
                    key='upload_file', on_change=clear_batch_pred) # clear when upload new
    with score_sample_cols[1]:
        st.markdown("Or load some sample pairs from the testing set")
        score_load_sample_btn = st.button(
            'Load Sample Article-summary pair', 
            on_click=score_load_sample, 
            type="secondary", 
            help = 'hello'
        )
    st.warning(body = "the csv file can either have two columns [article, target] or only 1 column article", icon = "📍")

    
    if upload_file is not None:
        #upload_articles = StringIO(upload_articles.getvalue().decode("utf-8"))
        upload_articles, upload_targets = read_batch_pred_articles(upload_file) # list of str
        st.session_state['upload_articles'] = upload_articles
        if upload_targets:
            # if target column appear as well
            st.session_state['upload_targets'] = upload_targets

    if 'upload_articles' in st.session_state:
        if 'upload_targets' in st.session_state:
            with st.expander(label = 'uploaded pairs', expanded = False):
                uploaded_pairs = pd.DataFrame({
                    'article' : st.session_state['upload_articles'], 
                    'target' : st.session_state['upload_targets']
                })
                st.table(uploaded_pairs)
        else:
            st.json(st.session_state['upload_articles'], expanded = False)

        btn_cols2 = st.columns([12, 6, 3, 5])
        with btn_cols2[0]:
            batch_pred_btn = st.button(
                'Batch Summarization', 
                on_click=batch_pred,
                key = "batch_pred_btn",
                kwargs=dict(articles=st.session_state['upload_articles']), 
                type="primary"
            )
        with btn_cols2[3]:
            if 'batch_pred_summ' in st.session_state:
                batch_pred_btn = st.download_button(
                    'Download', data = write_batch_pred(st.session_state['batch_pred_summ']), 
                    file_name='summarizations.txt', mime='text/txt', 
                    #on_click=batch_pred, kwargs=dict(articles=upload_articles)
                )
        with btn_cols2[1]:
            if 'upload_targets' in st.session_state and 'batch_pred_summ' in st.session_state:
                batch_send_to_arize = st.checkbox(
                    label = "Send to Arize?",
                    value = False,
                    key = "batch_send_to_arize"
                )

        with btn_cols2[2]:
            if 'upload_targets' in st.session_state and 'batch_pred_summ' in st.session_state:
                
                batch_log_btn = st.button(
                    'Log', 
                    on_click = batch_log, 
                    key = "batch_log_btn",
                    kwargs = dict(
                        articles = st.session_state['upload_articles'], 
                        summs = st.session_state['batch_pred_summ'], 
                        targets = st.session_state['upload_targets'], 
                        model_source = MODEL_CORE,
                        send_arize = batch_send_to_arize
                    ),
                    type = "secondary"
                )

        if 'batch_pred_err' in st.session_state:
            st.error(st.session_state['batch_pred_err'])

        if 'upload_targets' in st.session_state and 'batch_pred_summ' in st.session_state:
            st.divider()
            st.subheader("Scoring")
            score_paris = pd.DataFrame({
                    'article' : st.session_state['upload_articles'], 
                    'summ' : st.session_state['upload_targets']
                })
            batch_score_btn = st.button(
                'Compute Rouge Score', 
                on_click=score, 
                key="batch_score_btn",
                kwargs=dict(pair=score_paris), 
                type="primary"
            )

            if 'scores' in st.session_state:
                scores = st.session_state['scores']
                score_cols = st.columns(4)
                with score_cols[0]:
                    st.metric(label="rouge1", value=f"{scores['rouge1']:.1%}" , delta="unigram", delta_color='off')
                with score_cols[1]:
                    st.metric(label="rouge2", value=f"{scores['rouge2']:.1%}", delta="bigram", delta_color='off')
                with score_cols[2]:
                    st.metric(label="rougeL", value=f"{scores['rougeL']:.1%}", delta="longest", delta_color='off')
                with score_cols[3]:
                    st.metric(label="rougeLsum", value=f"{scores['rougeLsum']:.1%}", delta="overall")
                #st.json(st.session_state['scores'])
            if 'score_err' in st.session_state:
                st.error(st.session_state['score_err'])
    