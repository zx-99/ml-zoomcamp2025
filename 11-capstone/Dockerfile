FROM python:3.8.10-slim

WORKDIR /app

# Install dependencies
COPY Pipfile Pipfile.lock .

RUN pip install pipenv && \
    pipenv install --system --deploy
    
COPY . .

EXPOSE 9696

CMD["python", "app.py"]