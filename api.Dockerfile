# Use a slim base image
FROM python:3.12-slim

# Set environment variables to reduce image size and silence warnings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install OS-level dependencies for common packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your app (code, model files, etc.)
COPY . .

# Expose port used by FastAPI
EXPOSE 8000

# Run the FastAPI app using uvicorn (recommended for production)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -f api.Dockerfile -t fastapi-backend .