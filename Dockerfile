# Stage 1: Builder
FROM python:3.12-slim AS builder

# Set up a working directory
WORKDIR /app

# Install build dependencies
RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install unzip \
  && wget -O pyf_weights.zip https://www.dropbox.com/scl/fi/xe9m7qxo57c8int5uuzr2/pyf_weights.zip?rlkey=dhsb0d4vprs2caa4e1mcfjvhu \
  && unzip pyf_weights.zip \
  && rm pyf_weights.zip \
  && rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime
FROM python:3.12-slim

# Set up a working directory
WORKDIR /app

COPY --from=builder /app/weights /app/weights

# Copy app, install dependencies and make script executable
COPY . .
RUN apt-get update && apt-get install ffmpeg -y \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu ultralytics \
  && chmod +x run_app.sh

# Expose the application port
EXPOSE 5000

# Command to run the Flask app
CMD ./run_app.sh