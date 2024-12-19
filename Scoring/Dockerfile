FROM python:3.8.5-slim-buster

WORKDIR / C:\Users\joshu\Downloads\ML-Challenge\Scoring

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "./Scoring.py"]
