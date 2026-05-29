# Use official Python lightweight image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app/src

# Set work directory
WORKDIR /app

# Install system dependencies (needed for compiling some python packages and sqlite)
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    tzdata \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to Asia/Kolkata for Indian markets
ENV TZ="Asia/Kolkata"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY . /app/

# Expose Streamlit port
EXPOSE 8502

# The default command will be overridden by docker-compose
CMD ["python", "-m", "trade_system", "live"]
