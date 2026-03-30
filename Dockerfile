FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY knowledge_base/ knowledge_base/

RUN pip install --no-cache-dir .

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/fraud-agent /usr/local/bin/fraud-agent
COPY --from=builder /app/src/ src/
COPY --from=builder /app/knowledge_base/ knowledge_base/

EXPOSE 8000 50051

CMD ["uvicorn", "fraud_agent.api.rest:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
