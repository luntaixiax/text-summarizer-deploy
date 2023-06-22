from typing import Dict, List
from fastapi import File, FastAPI, UploadFile
import aiohttp
from pydantic import BaseModel
import pandas as pd
import evaluate
import os

if 'MODEL_URI' in os.environ:
    MODEL_URI = os.environ['MODEL_URI']
    if MODEL_URI == 'host':
        MODEL_URI = '172.17.0.1'  # docker access host ip
else:
    MODEL_URI = 'localhost'


app = FastAPI() # instantiate
rouge_score = evaluate.load("rouge")

@app.get("/")
def hello():
    return {"message" : "Welcome to CNN-Summarizer Model hub (Backend)"}

class Article(BaseModel):
    article: str

class Articles(BaseModel):
    articles: List[str]

class ArtRefPair(BaseModel):
    articles: List[str]
    targets: List[str]


@app.post("/article/summarize")  # path binding and HTTP method
async def online_pred(article: Article) -> str:
    article = article.dict()
    r = await batch_pred(articles=Articles(**{'articles' : [article['article']]}))
    return r[0]
            
@app.post("/article/summarize_batch")  # path binding and HTTP method
async def batch_pred(articles: Articles) -> List[str]:
    articles = articles.dict()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url = f'http://{MODEL_URI}:5003/invocations',
            json = {
                "dataframe_split": {
                    "columns": ["article"],
                    "data": [
                        [a] for a in articles['articles']
                    ]
                }
            },
            headers = {"Content-type": "application/json"}
        ) as p:
            r = await p.json()
    return [t['summarization'] for t in r['predictions']]

@app.post("/model/score")  # path binding and HTTP method
async def score(pairs: ArtRefPair) -> Dict[str, float]:
    pairs = pairs.dict()
    generated_summary = await batch_pred(
        articles = Articles(**{'articles' : pairs['articles']})
    )
    scores = rouge_score.compute(
            predictions=generated_summary, 
            references=pairs['targets'],
            use_aggregator = True
        )
    return scores