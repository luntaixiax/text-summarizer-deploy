from datetime import timedelta
import json
import os
from typing import List, Tuple
import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.mysql_operator import MySqlOperator
from airflow.hooks.base import BaseHook
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

import pandas as pd
from scripts.etl import extract, transform, load, \
    get_create_table_sql, get_insert_data_sql, get_load_data_sql, get_complete_message

@dag(
    schedule="@daily",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=["luntaixia"],
    dagrun_timeout=timedelta(minutes=60), # to prevent error
)
def cnn_daily():
    """
    ### CNN daily mail batch prediction DAG
    1. extract CNN daily mails using web scraper
    2. transform (summarize) CNN daily mails
    3. load and save data to mysql
    4. send complete message to slack channel
    """
    @task()
    def extract_() -> pd.DataFrame:
        """
        #### Extract task
        """
        df = extract(num_articles = 10)
        return df
    
    @task(multiple_outputs=False)
    def transform_(df: pd.DataFrame) -> pd.DataFrame:
        """
        #### Transform task
        """
        model_uri = Variable.get("model_uri")
        print("model_uri: ", model_uri)
        #os.makedirs('/opt/airflow/files', exist_ok=True)

        # csv_file_path = transform(
        #     model_uri=model_uri, 
        #     df=df,
        #     file_folder_path="/opt/airflow/files"
        # )
        df = transform(
            model_uri=model_uri, 
            df=df,
            #file_folder_path="/opt/airflow/files"
        )
        return df
    
    @task()
    def load_(df: pd.DataFrame) -> pd.DataFrame:
        """
        #### Load task
        """
        connection = BaseHook.get_connection("mysql_conn")
        print(connection.get_uri())
        load(
            df = df,
            conn_str=connection.get_uri()
        )

    @ task
    def get_stat(df: pd.DataFrame) -> str:
        return get_complete_message(df)

    # @task()
    # def convert_pd_to_list_tps(df: pd.DataFrame) -> List[Tuple]:
    #     return list(df.itertuples(index=False, name=None))

    df = extract_()
    transformed_df = transform_(df)
    
    #converted_df = convert_pd_to_list_tps(scored_df)

    create_table = MySqlOperator(
        sql=get_create_table_sql(), 
        task_id="CreateTable", 
        mysql_conn_id="mysql_conn",
    )

    # insert_data = MySqlOperator( 
    #     task_id='InsertData', 
    #     mysql_conn_id='mysql_conn',
    #     sql=get_insert_data_sql(), 
    #     parameters=converted_df, 
    #     #provide_context=True, 
    # )

    # load_ = MySqlOperator( 
    #     task_id='load_', 
    #     mysql_conn_id='mysql_conn',
    #     sql=get_load_data_sql(csv_file_path = csv_file_path),
    #     #provide_context=True, 
    # )

    msg = get_stat(transformed_df)

    send_slack_notification = SlackWebhookOperator(
        task_id = "send_slack_notification",
        http_conn_id = "slack_conn",
        message = msg,
        channel = "#monitoring"
    )

    transformed_df >> create_table >> load_(transformed_df) >> msg >> send_slack_notification
    
cnn_daily()