import os
import numpy as np
from functools import partial
import nltk
nltk.download('punkt')
import string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
import argparse

parser = argparse.ArgumentParser()
# model configs
parser.add_argument('--HF_DATA_ID', required=False, default='cnn_dailymail', type=str)
parser.add_argument('--HF_DATA_VER', required=False, default='3.0.0', type=str)
parser.add_argument('--MODEL_CHECKPOINT', required=False, default='t5-small', type=str)
# hyperparameters
parser.add_argument('--TRAIN_SIZE', required=False, default=100000, type=int)
parser.add_argument('--EVAL_SIZE', required=False, default=500, type=int)
parser.add_argument('--MAX_INPUT_LENGTH', required=False, default=512, type=int)
parser.add_argument('--MAX_TARGET_LENGTH', required=False, default=64, type=int)
parser.add_argument('--BATCH_SIZE', required=False, default=16, type=int)
parser.add_argument('--LR', required=False, default=0.01, type=float)
parser.add_argument('--WEIGHT_DECAY', required=False, default=4e-5, type=float)
parser.add_argument('--EPOCHS', required=False, default=2, type=int)
# training configs
parser.add_argument('--HAS_GPU', required=False, default=True, type=bool)
parser.add_argument('--LOG_STEPS', required=False, default=50, type=int)
parser.add_argument('--EVAL_STEPS', required=False, default=200, type=int)
parser.add_argument('--OUTPUT_DIR', required=True, type=str)
parser.add_argument('--HF_TOKEN', required=False, default = None, type=str)
parser.add_argument('--HF_HUB_REPO', required=False, default = None, type=str)

args = parser.parse_args()

# MLflow tracking settings
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'cnn-summarizer-fine-tune'
os.environ['MLFLOW_FLATTEN_PARAMS'] = '1'
os.environ['HF_MLFLOW_LOG_ARTIFACTS '] = 'True' # whether to copy artifacts

HAS_GPU = args.HAS_GPU
HF_DATA_ID = args.HF_DATA_ID
HF_DATA_VER = args.HF_DATA_VER
MODEL_CHECKPOINT = args.MODEL_CHECKPOINT
OUTPUT_DIR = args.OUTPUT_DIR

TRAIN_SIZE = args.TRAIN_SIZE
EVAL_SIZE = args.EVAL_SIZE
EVAL_STEPS = args.EVAL_STEPS
LOG_STEPS = args.LOG_STEPS
PREFIX = "summarize: "

### hyper parameters
MAX_INPUT_LENGTH = args.MAX_INPUT_LENGTH
MAX_TARGET_LENGTH = args.MAX_TARGET_LENGTH
BATCH_SIZE = args.BATCH_SIZE
LR = args.LR
WEIGHT_DECAY = args.WEIGHT_DECAY
EPOCHS = args.EPOCHS

HF_TOKEN = args.HF_TOKEN
HF_HUB_REPO = args.HF_HUB_REPO

def clean_text(text):
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                    if len(sent) > 0 and
                                    sent[-1] in string.punctuation]
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned

def preprocess_data(examples, tokenizer):
    texts_cleaned = [clean_text(text) for text in examples["article"]]
    inputs = [PREFIX + text for text in texts_cleaned]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    #with tokenizer.as_target_tokenizer():
    #  labels = tokenizer(examples["highlights"], max_length=max_target_length, 
    #                     truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


rouge_score = evaluate.load("rouge")

def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True) # token -> words
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) # token -> words
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    #return {k: round(v, 4) for k, v in result.items()}
    return result


# data processing and load pre-trained model
cnn_ds = load_dataset(HF_DATA_ID, HF_DATA_VER)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

preprocessor = partial(preprocess_data, tokenizer = tokenizer)
cnn_ds_cleaned = cnn_ds.filter(lambda example: (len(example['article']) >= 500) and (len(example['highlights']) >= 20))
cnn_ds_train = cnn_ds_cleaned['train'].shuffle(seed = 42).select(range(TRAIN_SIZE)).map(preprocessor, batched=True)
cnn_ds_valid = cnn_ds_cleaned['validation'].shuffle(seed = 42).select(range(EVAL_SIZE)).map(preprocessor, batched=True)

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

training_args = Seq2SeqTrainingArguments(
    output_dir = OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    logging_strategy="steps",
    logging_steps=LOG_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=3,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    fp16=HAS_GPU, # use 16-bit precision, only for gpu
    #load_best_model_at_end=True,
    #metric_for_best_model="rouge1",
    report_to="all",  # mlflow
    push_to_hub=False,
    no_cuda=not HAS_GPU
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=cnn_ds_train,
    eval_dataset=cnn_ds_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics= partial(compute_metrics, tokenizer = tokenizer)
)

trainer.train()
trainer.save_model()

if HF_TOKEN is not None:
    from huggingface_hub import login

    login(token = HF_TOKEN)
        
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    config = AutoConfig.from_pretrained(OUTPUT_DIR)
    model.push_to_hub(HF_HUB_REPO)
    tokenizer.push_to_hub(HF_HUB_REPO)