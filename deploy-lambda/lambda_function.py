import os
os.environ['JOBLIB_MULTIPROCESSING'] = "0"
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface/hub'  # lambda is read only and only tmp folder is writable
os.environ['HF_HOME'] = '/tmp/huggingface/hub'
import json
import pandas as pd
from typing import Dict
import pandas as pd
import nltk
nltk.download('punkt', download_dir = '/tmp/nltk_cache')
nltk.data.path.append('/tmp/nltk_cache')
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
print("Suucess import transformer")

import evaluate
from evaluate.utils.file_utils import DownloadConfig
DownloadConfig.cache_dir = '/tmp'
rouge_score = evaluate.load("rouge", cache_dir='/tmp', download_config=DownloadConfig(cache_dir = '/tmp'))
print("Successly imported Rouge Score from evaluate")


class CNNSummarizer:

    def __init__(self) -> None:
        #model_checkpoint = context.artifacts['model_checkpoint']
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./model"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "./model",
            return_dict=True
        )

    def online_predict(self, text: str, num_beans:int = 8, temperature: float = 1.0) -> str:

        text = "summarize: " + text
        tokened = self.tokenizer(
            text, max_length=512, truncation=True, return_tensors="pt"
        )
        s = self.model.generate(
            **tokened, 
            num_beams=num_beans, do_sample=True, temperature=temperature,
            min_length=10, max_length=64
        )
        decoded = self.tokenizer.decode(
            s[0],
            skip_special_tokens=True
        )
        first_sent = nltk.sent_tokenize(decoded.strip())[0]
        return first_sent
    
    def batch_predict(self, model_input: pd.Series, num_beans:int = 8, temperature: float = 1.0) -> pd.Series:
        kws = dict(num_beans=num_beans, temperature=temperature)
        return model_input.apply(self.online_predict, **kws)
    
    def score(self, sample_ref_pair: pd.DataFrame, num_beans:int = 8, temperature: float = 1.0) -> Dict[str, float]:
        """calculate ROUGE SCORE based on given sample-reference pair

        Args:
            sample_ref_pair (pd.DataFrame): contain two columns [articles, targets] where article is the input text 
                and targets is corresponding human-generated title for reference
            num_beans (int, optional): _description_. Defaults to 8.
            temperature (float, optional): _description_. Defaults to 1.0.

        Returns:
            Dict[str, float]: rouge scores (1, 2, L, Lsum)
        """

        generated_summary = self.batch_predict(
            sample_ref_pair['articles'],
            num_beans=num_beans,
            temperature=temperature
        )
        scores = rouge_score.compute(
            predictions=generated_summary, 
            references=sample_ref_pair['targets'],
            use_aggregator = True
        )
        return scores

summarizer = CNNSummarizer()



def lambda_handler(event:dict, context):
    """event should include ['action'], optional ['num_beans', 'temperature']
    if action = online-predict,
        event should be:
        {
            'action' : 'online-predict',
            'article' : 'abcdefg',
            'num_beans' : 8,  # optional
            'temperature' : 1.0 # optional
        }
    if action = batch-predict,
        event should be:
        {
            'action' : 'batch-predict',
            'articles' : ['abcdefg', 'hijklmn'],
            'num_beans' : 8,  # optional
            'temperature' : 1.0 # optional
        }
    if action = score,
        event should be:
        {
            'action' : 'score',
            'articles' : ['abcdefg', 'hijklmn'],
            'targets' : ['A', 'B'],
            'num_beans' : 8,  # optional
            'temperature' : 1.0 # optional
        }
    """
    

    if event['action'] == 'online-predict':
        text = event['article'] # str
        num_beans = event.get("num_beans", 8)
        temperature = event.get("temperature", 1.0)
        summ = summarizer.online_predict(
            text = text, 
            num_beans = num_beans, 
            temperature = temperature
        )
        return summ  # str

    elif event['action'] == 'batch-predict':
        texts = event['articles'] # list of str
        num_beans = event.get("num_beans", 8)
        temperature = event.get("temperature", 1.0)
        summ = summarizer.batch_predict(
            pd.Series(texts), 
            num_beans=num_beans, 
            temperature=temperature
        ).to_list()
        return summ  # list of str
    
    elif event['action'] == 'score':
        articles = event['articles'] # []
        targets = event['targets'] #  []
        num_beans = event.get("num_beans", 8)
        temperature = event.get("temperature", 1.0)
        summ = summarizer.score(
            pd.DataFrame({'articles': articles, 'targets' : targets}), 
            num_beans=num_beans, 
            temperature=temperature
        ) # dict of rouge scores
        return summ

    else:
        return "Please provide the valide parameter"
    

def api_gateway_lambda_handler(event:dict, context):
    """event should include ['rawPath', 'body']
        rawPath can be one of the following:
        1. /online-predict
        2. /batch-predict
        3. /score
    """

    call = {
        'action' : event['rawPath'].replace("/", ""),
    }
    params = json.loads(event['body'])
    call.update(params)
    print(call)
    return lambda_handler(event = call, context = context)