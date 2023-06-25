from typing import Dict, List
from fastapi import File, FastAPI, UploadFile
from pydantic import BaseModel
import pandas as pd
from apis import get_sample_pair, get_sample_pairs, get_pred_hist, get_pred_stat, \
    log_summs, log_summ, get_score_ts, init_db
from datetime import datetime

app = FastAPI()  # instantiate


@app.get("/")
def hello():
    return {"message": "Welcome to Monitoring Hub"}


@app.post("/db/init")  # path binding and HTTP method
def init_mysql():
    init_db()


class Logs(BaseModel):
    articles: List[str]
    summs: List[str]
    targets: List[str]
    model_source: str = 'Other'
    send_arize: bool = False


@app.post("/log/batch")  # path binding and HTTP method
def log_batch(logs: Logs) -> int:
    logs = logs.dict()
    return log_summs(
        articles=logs['articles'],
        summs=logs['summs'],
        targets=logs['targets'],
        model_source=logs['model_source'],
        send_arize=logs['send_arize']
    )


class Log(BaseModel):
    article: str
    summ: str
    target: str
    model_source: str = 'Other'
    send_arize: bool = False


@app.post("/log/online")  # path binding and HTTP method
def log_online(log: Log) -> int:
    log = log.dict()
    return log_summ(
        article=log['article'],
        summ=log['summ'],
        target=log['target'],
        model_source=log['model_source'],
        send_arize=log['send_arize']
    )


@app.get("/sample/pair")
def get_sample_pair_() -> dict:
    article, summ = get_sample_pair()
    return {
        'article': article,
        'summ': summ
    }


@app.get("/sample/pairs")
def get_sample_pairs_(num_sample: int = 10) -> dict:
    pairs = get_sample_pairs(num_sample=num_sample)
    return pairs.to_dict(orient='list')  # {article: [], summ: []}


@app.get("/history/list")
def list_history(num_record: int) -> list:
    hist = get_pred_hist(num_record=num_record)
    return hist.to_dict(orient='records')  # [{}, {}]


@app.get("/history/count")
def count_history(cur_ts: str, last_ts: str, freq: str = 'Day') -> list:
    stat = get_pred_stat(
        cur_ts=datetime.strptime(cur_ts, "%Y-%m-%d %H:%M:%S"),
        last_ts=datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S"),
        freq=freq
    )
    return stat.to_dict(orient='records')  # [{}, {}]


@app.get("/history/score")
def score_history(cur_ts: str, last_ts: str, freq: str = 'Day') -> list:
    stat = get_score_ts(
        cur_ts=datetime.strptime(cur_ts, "%Y-%m-%d %H:%M:%S"),
        last_ts=datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S"),
        freq=freq
    )
    return stat.to_dict(orient='records')  # [{}, {}]
