import newspaper
import pandas as pd
from tqdm.auto import tqdm
import requests
import os
from typing import List, Tuple
import datetime
#import mysql.connector
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, Session, Query

os.environ["no_proxy"]="*"

def post_req(url:str, params = None, data: dict = None) -> requests.Response:
    headers = {
        "Content-type": "application/json",
    }
    try:
        r = requests.post(url, params = params, json=data, headers=headers)
    except Exception as e:
        print("error happen here:\n", e)
    else:
        if r.status_code == 200:
            return r
        else:
            print("request code is not 200")
        
def get_req(url:str, params = None) -> requests.Response:
    headers = {"Content-type": "application/json"}
    try:
        r = requests.get(url, params = params, headers=headers)
    except Exception as e:
        print("error happen here:\n", e)
    else:
        if r.status_code == 200:
            return r.json()
        else:
            print("request code is not 200")

def extract(num_articles: int = 25) -> pd.DataFrame:
    link = 'https://www.cnn.com'
    # Scans the webpage and finds all the links on it.
    page_features = newspaper.build(
        link, 
        language='en', 
        memoize_articles=False
    )

    articles = tqdm(page_features.articles[:num_articles])
    data = []
    for article in articles:
        try:
            # Each article must be downloaded, then parsed individually.
            # This loads the text and title from the webpage to the object.
            article.download()
            article.parse()

            if not article.url.startswith('https://edition.cnn.com'):
                # Keep the text, title and URL from the article and append to a list.
                data.append({
                    'title':article.title,
                    'article':article.text,
                    'url': article.url})
                print(f"Successful get {article.title}")
        except Exception as e:
            # If, for any reason the download fails, continue the loop.
            print("Article Download Failed: " + str(e))

    df = pd.DataFrame.from_dict(data)
    print(df)
    df = df[df['article'].str.len() < 10000]
    return df

def transform(model_uri: str, df: pd.DataFrame) -> pd.DataFrame:
# def transform(model_uri: str, df: pd.DataFrame, file_folder_path: str) -> str:
    #df = df.sample(10)  # TODO: delete this
    # generate text summarization
    r = post_req(
        url = f"http://{model_uri}:8000/article/summarize_batch", 
        data = dict(
            articles=dict(articles=df['article'].tolist()),
            config=dict(num_beans=8, temperature=1.0)    
        )
    )
    df.loc[:, 'summary'] = r.json()
    #filename = os.path.join(file_folder_path, f"data-{datetime.datetime.now()}.csv")
    #df.to_csv(filename, index = False)
    #return filename
    return df

def get_create_table_sql() -> str:
    with open("/opt/airflow/dags/scripts/create_table.sql") as obj:
        return obj.read()

def get_insert_data_sql() -> str:
    with open("/opt/airflow/dags/scripts/insert_data.sql") as obj:
        return obj.read()

def get_load_data_sql(csv_file_path: str) -> str:
    with open("/opt/airflow/dags/scripts/load_data.sql") as obj:
        return obj.read().format(csv_file_path = csv_file_path)
    
def load(df: pd.DataFrame, conn_str:str):
    # with mysql.connector.connect(
    #     **connect_kws
    # ) as conn:
    #     with conn.cursor() as cursor:
    #         mysql_quey = """
    #         INSERT INTO summarizer.batch_summarization (title, article, `url`, summary) 
    #         VALUES (%s, %s, %s, %s)
    #         """
    #         params = list(df.itertuples(index=False, name=None))
    #         cursor.executemany(mysql_quey, params)
    #         conn.commit()
    engine = create_engine(conn_str)

    with engine.begin() as conn:
        df.to_sql(
            name = 'batch_summarization', 
            con = conn, 
            schema = 'summarizer',
            if_exists = 'append',
            index = False,
            method = 'multi',
            chunksize = 1000
        )

def get_complete_message(df: pd.DataFrame) -> str:
    num = len(df)
    avg_len_article = df['article'].str.len().mean()
    avg_len_summary = df['summary'].str.len().mean()
    msg = f"Successfully saved {num} CNN news, average length: article = {avg_len_article:.0f}, summary = {avg_len_summary:.0f} chars"
    return msg

if __name__ == '__main__':
    e = extract()
    t = transform(model_uri = 'localhost', df = e)
    print(t)


