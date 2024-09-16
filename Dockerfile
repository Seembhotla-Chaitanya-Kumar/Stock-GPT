FROM python:3.12.3-slim

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY stock-gpt.py .

ENTRYPOINT [ "streamlit","run","stock-gpt.py" ]