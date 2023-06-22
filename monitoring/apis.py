import tomli
import numpy as np
import pandas as pd
from typing import List, Tuple
import uuid
import datetime
from arize.pandas.generative.llm_evaluation import sacre_bleu, rouge
from arize.pandas.embeddings import EmbeddingGenerator, UseCases
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments, Schema, Metrics
from arize.utils.types import Environments, ModelTypes, EmbeddingColumnNames, Schema

from model import Base
from db.dbapi import dbIO
from db.dbconfigs import MySQL

with open(".secrets/secrets.toml", mode="rb") as fp:
    config = tomli.load(fp)

MYSQL_CONF = MySQL()
MYSQL_CONF.bindServer(config['mysql']['ip'], config['mysql']['port'], config['mysql']['db'])
MYSQL_CONF.login(config['mysql']['username'], config['mysql']['password'])
MYSQL_CONF.launch()
MYSQL_DB = dbIO(MYSQL_CONF)

EMBEDDER = EmbeddingGenerator.from_use_case(
    use_case=UseCases.NLP.SUMMARIZATION,
    model_name="distilbert-base-uncased",
    tokenizer_max_length=512,
    batch_size=100
)

ARIZE_CLIENT = Client(
    space_key = config['arize']['SPACE_KEY'],
    api_key = config['arize']['API_KEY']
)

def init_db():
    Base.metadata.create_all(MYSQL_CONF.engine)

def get_sample_pair() -> Tuple[str, str]:
    sql = """
    select 
        article, summ
    from 
        summarizer.summarize_sample 
    order by 
        rand() 
    limit 1
    """
    sample = MYSQL_DB.query_sql_df(sql)
    return sample.loc[0, 'article'], sample.loc[0, 'summ']

def get_sample_pairs(num_sample: int = 10) -> pd.DataFrame:
    sql = f"""
    select 
        article,
        summ
    from 
        summarizer.summarize_sample 
    order by 
        rand() 
    limit {num_sample}
    """
    samples = MYSQL_DB.query_sql_df(sql)
    return samples

def calc_score(df: pd.DataFrame) -> pd.DataFrame:
    df['score_blue'] = sacre_bleu( 
        response_col=df["summary"], 
        references_col=df["reference_summary"]
    )
    rouge_scores = rouge(
        response_col=df["summary"],
        references_col=df["reference_summary"],
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    for rouge_type, scores in rouge_scores.items():
        df[f"score_{rouge_type}"] = scores
    return df

def get_pred_hist(num_record: int) -> pd.DataFrame:
    sql = f"""
    select * 
    from summarizer.summarize_log 
    order by prediction_ts desc limit {num_record}
    """
    samples = MYSQL_DB.query_sql_df(sql)
    return samples.replace({np.nan:None})

def get_pred_stat(cur_ts: datetime.datetime, last_ts: datetime.datetime, freq:str = 'Day') -> pd.DataFrame:
    dt_fmt = {
        "Hour" : r"%Y-%m-%d %H:00:00",
        "Day" : r"%Y-%m-%d",
        "Month" : r"%Y-%m"
    }.get(freq)
    
    sql = f"""
    select
        model_source,
        date_format(prediction_ts, "{dt_fmt}") as prediction_dt,
        count(*) as num_record
    from 
        summarizer.summarize_log
    where
        prediction_ts between '{last_ts}' and '{cur_ts}'
    group by
        model_source,
        date_format(prediction_ts, "{dt_fmt}")
    order by
        model_source, prediction_dt desc
    """
    samples = MYSQL_DB.query_sql_df(sql)
    return samples

def get_score_ts(cur_ts: datetime.datetime, last_ts: datetime.datetime, freq:str = 'Day') -> pd.DataFrame:
    dt_fmt = {
        "Hour" : r"%Y-%m-%d %H:00:00",
        "Day" : r"%Y-%m-%d",
        "Month" : r"%Y-%m"
    }.get(freq)

    sql = f"""
    SELECT
        date_format(prediction_ts, "{dt_fmt}") as prediction_dt,
        count(*) as cnt,
        MIN(score_blue) as min_blue,
        AVG(score_blue) as score_blue,
        MAX(score_blue) as max_blue,
        STDDEV_SAMP(score_blue) as std_blue,
        MIN(score_rouge1) as min_rouge1,
        AVG(score_rouge1) as score_rouge1,
        MAX(score_rouge1) as max_rouge1,
        STDDEV_SAMP(score_rouge1) as std_rouge1,
        MIN(score_rouge2) as min_rouge2,
        AVG(score_rouge2) as score_rouge2,
        MAX(score_rouge2) as max_rouge2,
        STDDEV_SAMP(score_rouge2) as std_rouge2,
        MIN(score_rougeL) as min_rougeL,
        AVG(score_rougeL) as score_rougeL,
        MAX(score_rougeL) as max_rougeL,
        STDDEV_SAMP(score_rougeL) as std_rougeL
    from 
        summarizer.summarize_log
    where
        prediction_ts between '{last_ts}' and '{cur_ts}'
    group by
        date_format(prediction_ts, "{dt_fmt}")
    order by
        prediction_dt desc
    """
    r = MYSQL_DB.query_sql_df(sql)
    return r.replace({np.nan:None})

def generate_embedding(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "document_vector"] = EMBEDDER.generate_embeddings(text_col=df["document"])
    df.loc[:, "summary_vector"] = EMBEDDER.generate_embeddings(text_col=df["summary"])
    return df

def send_to_arize(df: pd.DataFrame) -> int:
    df = generate_embedding(df)
    prompt_columns=EmbeddingColumnNames(
        vector_column_name="document_vector",
        data_column_name="document"
    )
    response_columns=EmbeddingColumnNames(
        vector_column_name="summary_vector",
        data_column_name="summary"
    )
    schema = Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_ts",
        tag_column_names=["score_blue", "score_rouge1", "score_rouge2", "score_rougeL"],
        prompt_column_names=prompt_columns,
        response_column_names=response_columns,
    )
    response = ARIZE_CLIENT.log(
        dataframe = df,
        schema = schema,
        model_id = config['arize']['MODEL_ID'],
        model_version = "1.0",
        model_type = ModelTypes.GENERATIVE_LLM,
        environment = Environments.PRODUCTION
    )
    if response.status_code == 200:
        print(f"✅ Successfully logged data to Arize!")
        return 1
    else:
        print(
            f'❌ Logging failed with status code {response.status_code} and message "{response.text}"'
        )
        return 0
    
def log_summs(articles: List[str], summs: List[str], targets: List[str], model_source:str = 'Other', send_arize: bool = False) -> int:
    df = pd.DataFrame({
        'document' : articles,
        'summary' : summs,
        'reference_summary' : targets
    })
    df['prediction_id'] = df['document'].apply(
        lambda x: str(uuid.uuid4()).split("-")[0]
    )
    df.loc[:, 'prediction_ts'] = datetime.datetime.now()
    df.loc[:, 'model_source'] = model_source

    # drop over-length samples
    df = df[
        (df['document'].str.len() < 10000) 
        & (df['summary'].str.len() < 1000)
        & (df['reference_summary'].str.len() < 1000)
    ].reset_index(drop=True)

    # scoring
    df = calc_score(df)

    MYSQL_DB.insert_pd_df(
        tablename = 'summarize_log', 
        df = df, 
        schema = 'summarizer'
    )
    status = 1

    if send_arize:
        status = send_to_arize(df)

    return status

def log_summ(article: str, summ: str, target: str, model_source:str = 'Other', send_arize: bool = False) -> int:
    return log_summs(
        articles = [article],
        summs = [summ],
        targets = [target],
        model_source = model_source,
        send_arize = send_arize
    )

if __name__ == '__main__':
    init_db()
    # samples = get_sample_pair()
    # print(samples)
    # log_summs(articles = samples['article'], summs = samples['summ'], targets= samples['summ'], model_source='Other', send_arize=True)