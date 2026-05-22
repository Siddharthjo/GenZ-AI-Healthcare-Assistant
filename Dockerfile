FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-create directories that Flask/SQLAlchemy need at runtime
RUN mkdir -p instance

EXPOSE 3000

# Use gunicorn in production; 2 workers, 120s timeout for ML startup
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:3000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-"]
