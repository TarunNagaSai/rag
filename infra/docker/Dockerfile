FROM python:3.14.6-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

COPY . .

#final stage
FROM python:3.14.6-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "advanced_rag.router.api:app", "--host", "0.0.0.0", "--port", "8000"]
