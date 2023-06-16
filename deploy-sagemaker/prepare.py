import os
import shutil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

MODEL_CHECKPOINT = 'luntaixia/cnn-summarizer'

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT,
    config = AutoConfig.from_pretrained(MODEL_CHECKPOINT),
)
tokenizer.save_pretrained('./model')

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_CHECKPOINT,
    return_dict=True,
)
model.save_pretrained('./model')