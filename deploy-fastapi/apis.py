import os
from typing import Dict
import pandas as pd
import nltk
nltk.download('punkt')
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
import evaluate

MODEL_CHECKPOINT = os.environ['HF_HUB_REPO']

rouge_score = evaluate.load("rouge")

class CNNSummarizer:

    def __init__(self) -> None:
        #model_checkpoint = context.artifacts['model_checkpoint']
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CHECKPOINT,
            config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_CHECKPOINT,
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