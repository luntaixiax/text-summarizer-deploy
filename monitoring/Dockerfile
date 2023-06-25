FROM python:3.9-slim

ENV PORT=9020
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# # initialize database setup
# RUN python apis.py

COPY start.sh start.sh
# Set the start.sh file to be executable
RUN chmod +x ./start.sh

EXPOSE ${PORT}

ENTRYPOINT [ "./start.sh" ]