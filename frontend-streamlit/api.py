from datetime import date, datetime, timedelta
from io import StringIO
import requests
import pandas as pd
import os
from typing import List, Dict, Tuple
import json
import pandas as pd

import tomli

with open(".streamlit/secrets.toml", mode="rb") as fp:
    SECRETS = tomli.load(fp)


class ModelException(Exception):
    pass


class NotSupportedFunctionError(Exception):
    pass


def post_req(url: str, params=None, data: dict = None) -> requests.Response:
    headers = {
        "Content-type": "application/json",
    }
    try:
        r = requests.post(url, params=params, json=data, headers=headers)
    except Exception as e:
        print("error happen here:\n", e)
        raise ModelException(f"🥹 {str(e)}")
    else:
        if r.status_code == 200:
            print("request code: 200 OK")
            return r
        else:
            print("request code is not 200")
            raise ModelException(
                "🥹 Model does not seem to be working at this time, try again later!")


def get_req(url: str, params=None) -> requests.Response:
    headers = {"Content-type": "application/json"}
    try:
        r = requests.get(url, params=params, headers=headers)
    except Exception as e:
        print("error happen here:\n", e)
        raise ModelException(f"🥹 {str(e)}")
    else:
        if r.status_code == 200:
            print("request code: 200 OK")
            return r.json()
        else:
            print("request code is not 200")
            raise ModelException(
                "🥹 Model does not seem to be working at this time, try again later!")


class HandlerMlflowBackend:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def summarize(self, article: str) -> str:
        url = f"{self.endpoint}/article/summarize"
        data = dict(article=article)
        r = post_req(url=url, data=data)
        return r.json()

    def summarize_batch(self, articles: List[str]) -> List[str]:
        url = f"{self.endpoint}/article/summarize_batch"
        data = dict(articles=articles)
        r = post_req(url=url, data=data)
        return r.json()

    def scores(self, articles: List[str], targets: List[str]) -> Dict[str, float]:
        url = f"{self.endpoint}/model/score"
        data = dict(articles=articles, targets=targets)
        r = post_req(url=url, data=data)
        return r.json()


class HandlerLocalBackend(HandlerMlflowBackend):
    def __init__(self, endpoint: str, num_beans: int = 8, temperature: float = 1.0) -> None:
        self.endpoint = endpoint
        self.num_beans = num_beans
        self.temperature = temperature
        self.config = dict(num_beans=num_beans, temperature=temperature)

    def summarize(self, article: str) -> str:
        url = f"{self.endpoint}/article/summarize"
        data = dict(article=dict(article=article), config=self.config)
        r = post_req(url=url, data=data)
        return r.json()

    def summarize_batch(self, articles: List[str]) -> List[str]:
        url = f"{self.endpoint}/article/summarize_batch"
        data = dict(articles=dict(articles=articles), config=self.config)
        r = post_req(url=url, data=data)
        return r.json()

    def scores(self, articles: List[str], targets: List[str]) -> Dict[str, float]:
        url = f"{self.endpoint}/model/score"
        data = dict(pairs=dict(articles=articles,
                    targets=targets), config=self.config)
        r = post_req(url=url, data=data)
        return r.json()


class HandlerLambdaBackend(HandlerMlflowBackend):
    def __init__(self, endpoint: str, num_beans: int = 8, temperature: float = 1.0) -> None:
        self.endpoint = endpoint
        self.num_beans = num_beans
        self.temperature = temperature
        self.config = dict(num_beans=num_beans, temperature=temperature)

    def summarize(self, article: str) -> str:
        url = f"{self.endpoint}/online-predict"
        data = dict(article=article, **self.config)
        r = post_req(url=url, data=data)
        return r.text

    def summarize_batch(self, articles: List[str]) -> List[str]:
        url = f"{self.endpoint}/batch-predict"
        data = dict(articles=articles, **self.config)
        r = post_req(url=url, data=data)
        return r.json()

    def scores(self, articles: List[str], targets: List[str]) -> Dict[str, float]:
        url = f"{self.endpoint}/score"
        data = dict(articles=articles, targets=targets, **self.config)
        r = post_req(url=url, data=data)
        return r.json()


class HandlerSageMakerBackend:
    def __init__(self, endpoint_name: str = None) -> None:
        import boto3

        secrets = SECRETS['sagemaker']

        self.endpoint_name = endpoint_name if endpoint_name is not None else secrets[
            'sm_endpoint_name']
        self.runtime = boto3.client(
            'runtime.sagemaker',
            aws_access_key_id=secrets['aws_access_key_id'],
            aws_secret_access_key=secrets['aws_secret_access_key'],
            region_name=secrets['region']
        )

    def summarize(self, article: str) -> str:
        prompt = {
            "articles": [article]
        }
        try:
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=json.dumps(prompt),
                ContentType="application/json"
            )
        except Exception as e:
            raise ModelException(f"🥹 {str(e)}")
        else:
            r = response["Body"].read()
            r = json.loads(r)
            return r[0]

    def summarize_batch(self, articles: List[str]) -> List[str]:
        raise NotSupportedFunctionError(
            "🥹 This Model Type does not support batch summarization, too expensive!")

    def scores(self, articles: List[str], targets: List[str]) -> Dict[str, float]:
        raise NotSupportedFunctionError(
            "🥹 This Model Type does not support batch summarization, too expensive!")

# def read_batch_pred_articles(buffer: StringIO) -> List[str]:
#     return buffer.readlines()

def read_batch_pred_articles(buf) -> Tuple[List[str], List[str]]:
    sample = pd.read_csv(buf)
    articles = sample['article'].tolist()
    if 'summ' in sample.columns:
        summs = sample['summ'].tolist()
    else:
        summs = None
    return articles, summs


def write_batch_pred(summs: List[str]) -> str:
    # s = StringIO()
    # for summ in summs:
    #     s.write("{}\n".format(summ))
    # return s
    return '\n'.join(summs)


def get_upload_pair(buf) -> pd.DataFrame:
    sample = pd.read_csv(buf, usecols=['article', 'summ'])
    return sample


MONITOR_CONFIG = SECRETS['monitoring']
MONITOR_ENDPOINT = f"http://{MONITOR_CONFIG['ip']}:{MONITOR_CONFIG['port']}"

def init_db():
    r = post_req(
        url=f"{MONITOR_ENDPOINT}/db/init",
    )
    
def get_sample_article() -> Tuple[str, str]:
    r = get_req(
        url=f"{MONITOR_ENDPOINT}/sample/pair",
        params=None
    )
    return r['article'], r['summ']


def get_sample_pair(num_sample: int = 10) -> pd.DataFrame:
    r = get_req(
        url=f"{MONITOR_ENDPOINT}/sample/pairs",
        params={'num_sample': num_sample}
    )
    return pd.DataFrame(r)


def get_pred_hist(num_record: int) -> pd.DataFrame:
    r = get_req(
        url=f"{MONITOR_ENDPOINT}/history/list",
        params={'num_record': num_record}
    )
    return pd.DataFrame(r)


def get_pred_stat(cur_ts: datetime, last_ts: datetime, freq: str = 'Day') -> pd.DataFrame:
    r = get_req(
        url=f"{MONITOR_ENDPOINT}/history/count",
        params={
            'cur_ts': cur_ts.strftime("%Y-%m-%d %H:%M:%S"),
            'last_ts': last_ts.strftime("%Y-%m-%d %H:%M:%S"),
            'freq': freq
        }
    )
    return pd.DataFrame.from_records(r)


def get_score_ts(cur_ts: datetime, last_ts: datetime, freq: str = 'Day') -> pd.DataFrame:
    r = get_req(
        url=f"{MONITOR_ENDPOINT}/history/score",
        params={
            'cur_ts': cur_ts.strftime("%Y-%m-%d %H:%M:%S"),
            'last_ts': last_ts.strftime("%Y-%m-%d %H:%M:%S"),
            'freq': freq
        }
    )
    return pd.DataFrame.from_records(r)


def log_summs(articles: List[str], summs: List[str], targets: List[str], model_source: str = 'Other', send_arize: bool = False) -> int:
    r = post_req(
        url=f"{MONITOR_ENDPOINT}/log/batch",
        data=dict(
            articles=articles,
            summs=summs,
            targets=targets,
            model_source=model_source,
            send_arize=send_arize
        )
    )
    return r.json()


def log_summ(article: str, summ: str, target: str, model_source: str = 'Other', send_arize: bool = False):
    r = post_req(
        url=f"{MONITOR_ENDPOINT}/log/online",
        data=dict(
            article=article,
            summ=summ,
            target=target,
            model_source=model_source,
            send_arize=send_arize
        )
    )
    return r.json()


if __name__ == '__main__':
    ENDPOINT = ''
    hlb = HandlerLambdaBackend(endpoint=ENDPOINT)
    print(hlb.summarize_batch(
        articles=["One of the key ideas behind why you use instruction style prompts is that you can steer the model towards what you define as your goal state output, vs what the model, a user, or a different engineer believes it is.",
                  "One of the key ideas behind why you use instruction style prompts is that you can steer the model towards what you define as your goal state output, vs what the model, a user, or a different engineer believes it is."]
    ))

    # articles = ["One of the key ideas behind why you use instruction style prompts is that you can steer the model towards what you define as your goal state output, vs what the model, a user, or a different engineer believes it is."]
    # url = f"{ENDPOINT}/batch-predict"
    # data = dict(articles=articles, num_beans=10, temperature=1.0)
    # headers = {"Content-type": "application/json"}
    # r = requests.post(url, params = None, json=data, headers=headers)
    # print(r.status_code)

    # print(hlb.scores(
    #     articles = ["One of the key ideas behind why you use instruction style prompts is that you can steer the model towards what you define as your goal state output, vs what the model, a user, or a different engineer believes it is."],
    #     targets = ["The model is much better than GPT-3 at understanding more complex prompt instructions in zero-shot scenarios."]
    #     ))

    # text = "One of the key ideas behind why you use instruction style prompts is that you can steer the model towards what you define as your goal state output, vs what the model, a user, or a different engineer believes it is. For our use case this just means we can use the prompt to help the model better understand what information to include in our summary, what format we want it in, extractive vs abstractive etc. If we were to provide a prompt such as “summary this document” our model doesn’t have much information around what would be a good output for us, and will rely on its idea of what a good output is. The flip to this would be if we provide a prompt such as “Extract exact key information from the provided text in 5 sentences that focuses on xyz”. You can see how we’ve provided a much deeper idea to the model of what a good output is for us (goal state output).GPT-4 is much better than GPT-3 at understanding more complex prompt instructions in zero-shot scenarios, or scenarios where we don’t provide exact input and output examples of the summaries of example documents. We’ll see in examples below that the prompts we can provide can be much more complex, contain multiple steps, have clear guidelines, and much more that GPT-3 simply struggles with. The model is also very good at understanding the idea of pretending to be certain roles, such as a creative writer, an engineer, a student etc. This role play idea is something that we initially weren’t too keen on, as the value add didn’t make much sense over better prompts or prompt examples. Through testing and even integrating it into production we’ve found this role play idea very useful for less experienced prompt engineers or giving the model a more generalized idea of what output goal state we want to steer towards."

    # import boto3
    # import json

    # runtime= boto3.client(
    #     'runtime.sagemaker',
    #     region_name='ca-central-1'
    # )

    # prompt = {
    #     "articles": [text]
    # }

    # response = runtime.invoke_endpoint(
    #     EndpointName='text-summarizer-2023-06-16-15-25-37',
    #     Body=json.dumps(prompt),
    #     ContentType="application/json"
    # )

    # r = response["Body"].read()
    # r = json.loads(r)
    # print(r[0])

    #print(get_pred_stat(cur_ts=datetime.now(), last_ts=datetime.now() - timedelta(days = 1), freq='Hour'))
    # r = log_summs(
    #     articles = ["The world's last surviving male northern white rhino - stripped of his horn for his own safety - is now under 24-hour armed guard in a desperate final bid to save the species. Sudan is guarded day and night by a group of rangers who risk their lives on a daily basis as they try to keep the rhino from poachers lured by the rising price of ivory. But even without his horn, keepers in the Kenyan reserve of Ol Pojeta in fear for his safety. Scroll down for video . Guard: The rangers keep an armed watch around Sudan at all times to deter poachers after his horn . Extreme measures: Rangers have even cut off the rhino's horn - but they fear it won't be enough . Hungry: Feeding time in Sudan's enclosure - he spent most of his life in a Czech zoo . Dangerous: The rangers are aware they are risking their lives to protect the enormous animal . The 43-year-old rhino - who could live until his 50s - is the last chance for any future northern white rhino calves. Sudan was moved, along with two female rhinos, from a zoo in the Czech Republic in December 2009. The reserve, which specialises in the conservation of rhinos, was chosen because of its successful breeding programme with black rhinos. It had been hoped the move would encourage them to breed, but all attempts have been unsuccessful. The project was dealt a further blow when Suni - the world's only other male, who also lived at Ol Pojeta - died last October. It left just five northern white rhinos in the world - and the three in Kenya are in particular danger. Hunted: Rhinos like Sudan have no predators in the wild because of their size - apart from humans . New home: It had been hoped moving Sudan to Kenya with two females would encourage them to breed . Simor Irungu, one of the rangers who guards Sudan, says the team regularly risk their lives to keep him safe. 'With the rising demand for rhino horn and ivory, we face many poaching attempts and while we manage to counter a large number of these, we often risk our lives in the line of duty.' It is a sad end for a species which used to roam across the heart of Africa - from southern Chad, across the Democratic Republic of Congo and up into Sudan. Just over half a century ago, there were 2,000 northern white rhinos; but 1984 there were only 15, all in the DRC, according to the World Wildlife Fund. But then conservationist managed to bring them back from the brink, and bought the population up to at least 30 animals less than a decade later. Failure: But attempts at breeding have been unsuccessful - and Sudan is now getting old . High value: The price of ivory is now said to be between £40,000 and £47,000 a kilo . But then poaching took its toll, and the entire park was emptied. The last northern white rhinos were spotted in 2006. Their extinction has been fueled by the growing demand for ivory, which comes in large part from the Far East, where it is believed to be a cure for several ailments. The price for ivory has risen from between £170 to £541 per kilo in the 1990s, to today's prices of £40,000 to £47,355 per kilo, according to a report by the International Fund for Animal Welfare. The rangers have taken steps to deter the poachers, but they still fear it may not be enough. Elodie Sampere explained: 'The only reason his horn has been cut off is to deter poachers. 'If the rhino has no horn, he is of no interest to them. 'This is purely to keep him safe.' Sad end: Sudan is one of five northern white rhinos left, and a number of them are his descedents . However, keeping them safe is a costly business: the team of 40 cost £75,000 for six months. It is usually paid for with money made from tourism, but recent instability in Kenya, and fear of Ebola - which is actually thousands of miles away - have kept people away. So the team at  Ol Pejeta is hoping to raise the money through crowd funding. 'Keeping the ranger team safe is expensive,' the appeal reads. 'They are given world-class training, and are kitted out with the latest in equipment and support, from night vision goggles to GPS tracking, to a team of tracking and support dogs.... 'Keeping the team funded and equipped is an ongoing challenge. 'We are aiming to raise enough to safeguard the wages for the forty strong team for the next six months. 'This is £75,000. Any which way, every single pound contributed will help secure the rangers, that secure the rhino, for us and for future generations.' To donate, visit Ol Pejeta's GoFundMe page."],
    #     summs = ["Sudan is the last hope for a species on the verge of being wiped out ."],
    #     targets = ["Sudan is the last hope for a species on the verge of being wiped out ."],
    #     model_source = "Other",
    #     send_arize = False
    # )
    # print(r)

    # r = log_summ(
    #     article = "The world's last surviving male northern white rhino - stripped of his horn for his own safety - is now under 24-hour armed guard in a desperate final bid to save the species. Sudan is guarded day and night by a group of rangers who risk their lives on a daily basis as they try to keep the rhino from poachers lured by the rising price of ivory. But even without his horn, keepers in the Kenyan reserve of Ol Pojeta in fear for his safety. Scroll down for video . Guard: The rangers keep an armed watch around Sudan at all times to deter poachers after his horn . Extreme measures: Rangers have even cut off the rhino's horn - but they fear it won't be enough . Hungry: Feeding time in Sudan's enclosure - he spent most of his life in a Czech zoo . Dangerous: The rangers are aware they are risking their lives to protect the enormous animal . The 43-year-old rhino - who could live until his 50s - is the last chance for any future northern white rhino calves. Sudan was moved, along with two female rhinos, from a zoo in the Czech Republic in December 2009. The reserve, which specialises in the conservation of rhinos, was chosen because of its successful breeding programme with black rhinos. It had been hoped the move would encourage them to breed, but all attempts have been unsuccessful. The project was dealt a further blow when Suni - the world's only other male, who also lived at Ol Pojeta - died last October. It left just five northern white rhinos in the world - and the three in Kenya are in particular danger. Hunted: Rhinos like Sudan have no predators in the wild because of their size - apart from humans . New home: It had been hoped moving Sudan to Kenya with two females would encourage them to breed . Simor Irungu, one of the rangers who guards Sudan, says the team regularly risk their lives to keep him safe. 'With the rising demand for rhino horn and ivory, we face many poaching attempts and while we manage to counter a large number of these, we often risk our lives in the line of duty.' It is a sad end for a species which used to roam across the heart of Africa - from southern Chad, across the Democratic Republic of Congo and up into Sudan. Just over half a century ago, there were 2,000 northern white rhinos; but 1984 there were only 15, all in the DRC, according to the World Wildlife Fund. But then conservationist managed to bring them back from the brink, and bought the population up to at least 30 animals less than a decade later. Failure: But attempts at breeding have been unsuccessful - and Sudan is now getting old . High value: The price of ivory is now said to be between £40,000 and £47,000 a kilo . But then poaching took its toll, and the entire park was emptied. The last northern white rhinos were spotted in 2006. Their extinction has been fueled by the growing demand for ivory, which comes in large part from the Far East, where it is believed to be a cure for several ailments. The price for ivory has risen from between £170 to £541 per kilo in the 1990s, to today's prices of £40,000 to £47,355 per kilo, according to a report by the International Fund for Animal Welfare. The rangers have taken steps to deter the poachers, but they still fear it may not be enough. Elodie Sampere explained: 'The only reason his horn has been cut off is to deter poachers. 'If the rhino has no horn, he is of no interest to them. 'This is purely to keep him safe.' Sad end: Sudan is one of five northern white rhinos left, and a number of them are his descedents . However, keeping them safe is a costly business: the team of 40 cost £75,000 for six months. It is usually paid for with money made from tourism, but recent instability in Kenya, and fear of Ebola - which is actually thousands of miles away - have kept people away. So the team at  Ol Pejeta is hoping to raise the money through crowd funding. 'Keeping the ranger team safe is expensive,' the appeal reads. 'They are given world-class training, and are kitted out with the latest in equipment and support, from night vision goggles to GPS tracking, to a team of tracking and support dogs.... 'Keeping the team funded and equipped is an ongoing challenge. 'We are aiming to raise enough to safeguard the wages for the forty strong team for the next six months. 'This is £75,000. Any which way, every single pound contributed will help secure the rangers, that secure the rhino, for us and for future generations.' To donate, visit Ol Pejeta's GoFundMe page.",
    #     summ = "Sudan is the last hope for a species on the verge of being wiped out .",
    #     target = "Sudan is the last hope for a species on the verge of being wiped out .",
    #     model_source = "Other",
    #     send_arize = False
    # )
    # print(r)
