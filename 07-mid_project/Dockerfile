FROM python:3.12.1-slim

# Create app directory
WORKDIR /app

# Install dependencies
COPY Pipfile Pipfile.lock ./

RUN pip install pipenv && \
    pipenv install --system --deploy

# Copy source code 
COPY . ./

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
