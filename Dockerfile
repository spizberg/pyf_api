FROM python:3.12-slim
RUN mkdir /app
WORKDIR /app
EXPOSE 5000
RUN  apt-get update \
  && apt-get install -y wget \
  && apt-get install unzip \
  && apt-get update && apt-get install ffmpeg libsm6 libxext6 -y \
  && wget -O pyf_weights.zip https://www.dropbox.com/scl/fi/xe9m7qxo57c8int5uuzr2/pyf_weights.zip?rlkey=dhsb0d4vprs2caa4e1mcfjvhu \
  && unzip pyf_weights.zip \
  && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install -r requirements.txt && chmod +x run_app.sh
CMD ./run_app.sh