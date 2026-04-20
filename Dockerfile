FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user:user requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY --chown=user:user . .

USER user

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HOME=/home/user \
    WEB_CONCURRENCY=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys,urllib.request; \
u=urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4); \
sys.exit(0 if u.status==200 else 1)"

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
