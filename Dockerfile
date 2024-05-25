FROM python:3.6.8

WORKDIR /app
EXPOSE 5000

COPY requirements.txt /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --upgrade pip \
    && pip3 install -r requirements.txt

COPY . /app

CMD ["uvicorn", "main:app" , "--host=0.0.0.0","--log-config=log-config.json"]