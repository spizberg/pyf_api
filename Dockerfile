FROM python:3.12-slim
RUN mkdir /app
WORKDIR /app
EXPOSE 5000
RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install unzip \
  && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install -r requirements.txt && chmod +x run_app.sh
CMD ./run_app.sh