FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./

RUN pip install --upgrade pip && pip install .

COPY src ./src
COPY entrypoint.sh ./entrypoint.sh
COPY .dvc/config ./.dvc/config
COPY dvc.yaml dvc.lock ./

RUN chmod +x ./entrypoint.sh \
    && git init && git config user.email "ci@local" && git config user.name "ci"

ENTRYPOINT ["./entrypoint.sh"]