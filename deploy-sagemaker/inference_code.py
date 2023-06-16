import os
import json
from typing import List
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
import nltk
nltk.download('punkt')

def model_fn(model_dir: str) -> dict:
    """
    Load the model for inference
    """

    model_path = os.path.join(model_dir, 'model')

    # load the config
    config = AutoConfig.from_pretrained(model_path)

    # Load tokenizer from disk.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        config = config
    )

    # Load model from disk.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    model_dict = {'model': model, 'tokenizer':tokenizer}

    return model_dict

def _core_predict(text: str, model: dict, num_beans:int = 8, temperature: float = 1.0) -> str:
    tokenizer = model['tokenizer']
    model = model['model']
    text = "summarize: " + text

    tokened = tokenizer(
        text, max_length=512, truncation=True, return_tensors="pt"
    )
    s = model.generate(
        **tokened, 
        num_beams=num_beans, do_sample=True, temperature=temperature,
        min_length=10, max_length=64
    )
    decoded = tokenizer.decode(
        s[0],
        skip_special_tokens=True
    )
    first_sent = nltk.sent_tokenize(decoded.strip())[0]
    return first_sent


def predict_fn(texts: List[str], model: dict) -> List[str]:
    """
    Apply model to the incoming request
    """
    summs = []
    for text in texts:
        summ = _core_predict(
            text = text,
            model = model
        )
        summs.append(summ)
    return summs


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    if request_content_type == "application/json":
        request = json.loads(request_body)
        print(request)
        return request['articles']
    else:
        raise TypeError("Only accept application/json format request")

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """

    if response_content_type == "application/json":
        return prediction
    else:
        raise TypeError("Only send application/json format response")