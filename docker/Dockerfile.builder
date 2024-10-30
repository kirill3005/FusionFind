FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Создание кэша:
COPY . /app # Копируем весь проект, чтобы кэширование работало