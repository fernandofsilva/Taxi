# Image base
FROM jupyter/scipy-notebook:latest

LABEL maintainer="Fernando Silva <fernando.f.silva@outlook.com>"

# Copy all files to app folder
COPY .. /app

# Set working directory
WORKDIR /app

## Upgrade pip
RUN pip3 install --upgrade pip

# Install
RUN pip install -r requirements.txt

# Run python script
ENTRYPOINT ["python", "/app/codes/main.py"]