FROM python:3.10

WORKDIR /usr/src/app

COPY *.py .
COPY simulation_requirements.txt .

RUN pip install --no-cache-dir -r simulation_requirements.txt
