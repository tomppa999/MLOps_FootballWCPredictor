FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src
COPY tests /app/tests
COPY docs /app/docs
COPY data_samples /app/data_samples

RUN pip install --upgrade pip && pip install .

CMD ["python", "-m", "src"]