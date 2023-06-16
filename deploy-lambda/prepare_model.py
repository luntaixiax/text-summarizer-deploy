import os
import shutil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

MODEL_CHECKPOINT = os.environ['HF_HUB_REPO']
print(MODEL_CHECKPOINT)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT,
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT, cache_dir = './cache'),
    cache_dir = './cache'
)
tokenizer.save_pretrained('./model')
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_CHECKPOINT,
    return_dict=True,
    cache_dir = './cache'
)
model.save_pretrained('./model')

shutil.rmtree("./cache")
print("Cleaned Cached Model Data!")