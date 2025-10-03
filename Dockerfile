# Small, stable base
FROM python:3.11-slim

# Keep Python quiet & logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Optional but helpful so "import backend" works from /app
ENV PYTHONPATH=/app

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Run as non-root
RUN useradd -m appuser
USER appuser

EXPOSE 8080

# Keep it simple: 1 worker, more threads to conserve RAM
# Use exec form so the process gets signals correctly
CMD ["gunicorn", "whatsapp_bot:app", \
     "-k", "gthread", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "8", \
     "--timeout", "60", \
     "--keep-alive", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
