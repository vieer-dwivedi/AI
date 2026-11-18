# Simple words: builds a small container that runs the webhook API.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent data (vector DB, reports) live in these folders - mount them as volumes.
RUN mkdir -p /app/data /app/out
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
