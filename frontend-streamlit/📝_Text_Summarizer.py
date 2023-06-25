import streamlit as st
from api import init_db, ModelException

def do_stuff_on_page_load():
    st.set_page_config(layout='wide')

def create_static_k_v_markdown(key: str, value: str, icon: str = None):
    cols = st.columns([1, 50])
    with cols[0]:
        st.image(icon, width = 20)
    with cols[1]:
        st.caption(key)
    st.markdown(f"""
    ```
    {value}
    ```                        
    """)

def init_db_():
    try:
        init_db()
    except ModelException as e:
        st.error(f"Failed: {e}")
    else:
        st.info("Successfully initialized database!")

do_stuff_on_page_load()

st.image(
    image = "https://w.wallhaven.cc/full/d6/wallhaven-d6eedl.png", 
    #caption="Summarizer",
    use_column_width='auto'
)

with st.expander(label = "Project Structure", expanded = False):
    st.components.v1.html("""
        <iframe 
            width="1800" height="1200" 
            src="https://miro.com/app/live-embed/uXjVM9PubHU=/?moveToViewport=-1885,-898,2316,1192&embedId=33851316624" 
            frameborder="0" scrolling="no" allow="fullscreen; 
            clipboard-read; 
            clipboard-write" allowfullscreen>
        </iframe>
        """,
        height = 1200,
        scrolling = True
    )

cols = st.columns(2)
with cols[0]:
    create_static_k_v_markdown(
        key = 'Training Dataset',
        value = "https://huggingface.co/datasets/cnn_dailymail",
        icon = "https://www.clipartmax.com/png/middle/219-2197837_training-icon-marketing.png"
    )
    create_static_k_v_markdown(
        key = 'Pretrained Model',
        value = "https://huggingface.co/t5-small",
        icon = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Istanbul_T5_Line_Symbol.svg/2048px-Istanbul_T5_Line_Symbol.svg.png"
    )
    create_static_k_v_markdown(
        key = 'Performance Metric',
        value = "Rouge Score",
        icon = "https://static.vecteezy.com/system/resources/thumbnails/010/311/019/small/speedometer-icon-on-white-background-colorful-gauge-sign-credit-score-meter-symbol-flat-style-vector.jpg"
    )

with cols[1]:
    create_static_k_v_markdown(
        key = 'Github Code Repo',
        value = "https://github.com/luntaixiax/text-summarizer-deploy",
        icon = "https://cdn-icons-png.flaticon.com/512/25/25231.png"
    )
    create_static_k_v_markdown(
        key = 'Model on Huggingface Hub',
        value = "https://huggingface.co/luntaixia/cnn-summarizer",
        icon = "https://icons.iconarchive.com/icons/microsoft/fluentui-emoji-3d/512/Hugging-Face-3d-icon.png"
    )
    create_static_k_v_markdown(
        key = 'Models on Docker Hub',
        value = "https://hub.docker.com/repository/docker/luntaixia/cnn-summarizer-mlflow/general",
        icon = "https://hub.docker.com/search?q=luntaixia"
    )

st.divider()
st.button(
    'Database Initialization', 
    on_click=init_db_,
    key = "init_db",
    type="secondary"
)