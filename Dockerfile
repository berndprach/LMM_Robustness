FROM python:3.13

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./data ./data
COPY ./src ./src

CMD ["python", "./src/gui.py", "8050"]
