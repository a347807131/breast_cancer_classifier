FROM python:3.6.8

WORKDIR /app
EXPOSE 8000

COPY requirements.txt /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --upgrade pip

RUN pip install torch==1.10.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY . /app

CMD ["uvicorn", "main:app" , "--host=0.0.0.0","--log-config=log-config.json"]