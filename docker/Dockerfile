FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get purge -y --auto-remove build-essential
WORKDIR /app