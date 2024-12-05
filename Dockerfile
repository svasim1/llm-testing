FROM python:3.12.7

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install gunicorn

COPY . .

WORKDIR /app/app

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4"]

# CMD ["tail", "-f", "/dev/null"] # Debug