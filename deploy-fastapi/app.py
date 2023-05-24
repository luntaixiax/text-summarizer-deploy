from typing import Dict, List
from fastapi import File, FastAPI, UploadFile
import uvicorn
from pydantic import BaseModel
import pandas as pd
from apis import summarizer

app = FastAPI() # instantiate

@app.get("/")
def hello():
    return {"message" : "Welcome to CNN-Summarizer Model hub"}

class Article(BaseModel):
    article: str

class Articles(BaseModel):
    articles: List[str]

class ArtRefPair(BaseModel):
    articles: List[str]
    targets: List[str]

class Config(BaseModel):
    num_beans: int = 8
    temperature: float = 1.0

@app.post("/article/summarize")  # path binding and HTTP method
def online_pred(article: Article, config: Config) -> str:
    article = article.dict()
    configs = config.dict()
    return summarizer.online_predict(
        article['article'], 
        num_beans=configs['num_beans'], 
        temperature=configs['temperature']
    )

@app.post("/article/summarize_batch")  # path binding and HTTP method
def batch_pred(articles: Articles, config: Config) -> List[str]:
    articles = articles.dict()
    configs = config.dict()
    return summarizer.batch_predict(
        pd.Series(articles['articles']), 
        num_beans=configs['num_beans'], 
        temperature=configs['temperature']
    ).to_list()

@app.post("/model/score")  # path binding and HTTP method
def score(pairs: ArtRefPair, config: Config) -> Dict[str, float]:
    pairs = pairs.dict() 
    configs = config.dict()
    return summarizer.score(
        pd.DataFrame(pairs), 
        num_beans=configs['num_beans'], 
        temperature=configs['temperature']
    )
