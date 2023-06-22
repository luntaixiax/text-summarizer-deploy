from datetime import datetime, timedelta
import streamlit as st
from api import get_pred_hist, get_pred_stat, get_score_ts
from chart_api import table_nonedit_general, chart_hist_stat, chart_hist_score

def do_stuff_on_page_load():
    st.set_page_config(layout='wide')

do_stuff_on_page_load()

def get_hist_chart_config(freq: str):
    # how long will be the history to be extracted
    length = {
        "Hour" : dict(hours = 24),
        "Day" : dict(days = 30),
        "Month" : dict(weeks = 52)
    }.get(freq)
    # how width will be one block
    width = {
        "Hour" : dict(minutes = 45),
        "Day" : dict(hours = 20),
        "Month" : dict(days = 25)
    }.get(freq)
    return length, width


freq = st.radio(label = "Stat Freq", options = ['Hour', 'Day', 'Month'], index = 0,
                horizontal = True)
length, width = get_hist_chart_config(freq = freq)

stat = get_pred_stat(cur_ts=datetime.now(), last_ts = datetime.now() - timedelta(**length), freq = freq)
chart = chart_hist_stat(hist_stat=stat, title = '# of historical predictions by model source', freq = freq)
st.bokeh_chart(chart, use_container_width=True)

metric = st.selectbox(label = "Choose Score", options = ['blue', 'rouge1', 'rouge2', 'rougeL'], index = 0)
scores = get_score_ts(cur_ts=datetime.now(), last_ts = datetime.now() - timedelta(**length), freq = freq)
score_chart = chart_hist_score(hist_score=scores, metric=metric,title = 'historical scores', freq = freq)
st.bokeh_chart(score_chart, use_container_width=True)

st.subheader("Latest Prediction")
hist = get_pred_hist(num_record = 10)
hist_table = table_nonedit_general(df = hist, index_col='prediction_id', dt_columns = ['prediction_ts'])
st.bokeh_chart(hist_table, use_container_width=True)