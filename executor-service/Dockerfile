FROM python:3.12-slim

# ---------- System ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---------- Install system deps ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- App ----------
COPY app ./app
COPY workdir ./workdir
COPY logs ./logs

# ---------- Default command ----------
CMD ["python", "-m", "app.main"]
