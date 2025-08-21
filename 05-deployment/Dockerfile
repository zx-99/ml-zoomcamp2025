FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

# 复制依赖定义文件
COPY Pipfile Pipfile.lock ./

# 安装pipenv并安装依赖
RUN pip install pipenv && \
    pipenv install --system --deploy

# Copy any python files and the model we had to the working directory of Docker 
COPY *.py *.bin ./

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "5-flask:app"]


