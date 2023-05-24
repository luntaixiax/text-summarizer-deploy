FROM python:3.9-slim

ENV PORT=5003
ENV HF_HUB_REPO="luntaixia/cnn-summarizer"

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip; \
    pip install --no-cache-dir -r requirements.txt

# RUN apt-get update && apt-get install -y curl git
# RUN curl https://pyenv.run | bash
# RUN python -m  pip install virtualenv
# RUN apt-get install -y libbz2-dev

COPY . .

RUN python build_model.py
#RUN rm -r cnn-summarizer

COPY start.sh start.sh

# Set the start.sh file to be executable
RUN chmod +x ./start.sh

EXPOSE ${PORT}

ENTRYPOINT [ "./start.sh" ]