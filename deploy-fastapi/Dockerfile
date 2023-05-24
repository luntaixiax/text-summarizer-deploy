FROM python:3.9-slim

ENV PORT=8000
ENV HF_HUB_REPO='luntaixia/cnn-summarizer'
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY start.sh start.sh
# Set the start.sh file to be executable
RUN chmod +x ./start.sh

EXPOSE ${PORT}

ENTRYPOINT [ "./start.sh" ]