FROM python:3.6.8

WORKDIR /app
EXPOSE 8000

COPY requirements.txt /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install -r requirements.txt

COPY . /app

CMD ["uvicorn", "main:app" , "--host=0.0.0.0","--log-config=log-config.json"]