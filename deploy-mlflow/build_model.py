import os
import mlflow
import pandas as pd
import nltk
nltk.download('punkt')

MODEL_CHECKPOINT = os.environ['HF_HUB_REPO']

class CNNSummarizer(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

        #model_checkpoint = context.artifacts['model_checkpoint']
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CHECKPOINT,
            config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_CHECKPOINT,
            return_dict=True
        )

    def summarize(self, text: str) -> str:

        text = "summarize: " + text
        tokened = self.tokenizer(
            text, max_length=512, truncation=True, return_tensors="pt"
        )
        s = self.model.generate(
            **tokened, 
            num_beams=8, do_sample=True, min_length=10, max_length=64
        )
        decoded = self.tokenizer.decode(
            s[0],
            skip_special_tokens=True
        )
        first_sent = nltk.sent_tokenize(decoded.strip())[0]
        return first_sent
    
    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        return model_input['article'].apply(self.summarize).rename('summarization')
    

if __name__ == '__main__':
    import json
    # with open("requirements.txt") as obj:
    #     reqs = obj.readlines()

    # Save the MLflow Model
    mlflow_pyfunc_model_path = "cnn-summarizer_mlflow_pyfunc"
    mlflow.pyfunc.save_model(
        path = mlflow_pyfunc_model_path, 
        python_model = CNNSummarizer(),
        # artifacts = {
        #     "model_checkpoint": "",
        # },
        signature = mlflow.models.ModelSignature.from_dict({
            'inputs': json.dumps([{'name' : 'article', 'type' : 'string'}]),
            'outputs': json.dumps([{'name' : 'summarization', 'type' : 'string'}])
        }),
        #pip_requirements="requirements.txt"
        #code_path="",
        # conda_env={
        #     'channels': ['defaults'],
        #     'dependencies': [
        #         'python={}'.format(3.9),
        #         'pip',
        #         {
        #             'pip': reqs,
        #         },
        #     ],
        #     'name': 'cmm-summarizer-env'
        # }
    )

    # load model -- testing and pre-download trained-model cache
    loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)
    print(loaded_model._model_meta.signature)
    mlflow.end_run()