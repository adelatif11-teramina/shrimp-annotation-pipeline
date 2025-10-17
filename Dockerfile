# Railway Deployment Dockerfile for Shrimp Annotation Pipeline
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build frontend (skip if fails)
RUN cd ui && npm install && npm run build || echo "Frontend build failed, continuing..."

# Create required directories
RUN mkdir -p logs data/exports data/feedback data/local

# Set environment variables for Railway
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV API_HOST=0.0.0.0
ENV DB_HOST=localhost
ENV DB_PORT=5432
ENV DB_NAME=annotations
ENV DB_USER=postgres
ENV DEBUG=false
ENV LOG_LEVEL=INFO

# Expose port (Railway will set PORT env var)
EXPOSE ${PORT:-8000}

# Copy and make startup script executable
COPY railway_direct_start.sh /app/
RUN chmod +x /app/railway_direct_start.sh

# Start application with startup script
CMD ["/app/railway_direct_start.sh"]