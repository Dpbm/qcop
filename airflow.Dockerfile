FROM python:3.12.10-slim-bullseye AS setup

COPY requirements.txt .
RUN python -m venv /proj-venv

ENV PIPENV="/proj-venv/bin/pip"

RUN ${PIPENV} install -r requirements.txt

FROM apache/airflow:slim-2.11.0-python3.12 AS serve

COPY --from=setup /proj-venv/lib/python3.12/site-packages/ /home/airflow/.local/lib/python3.12/site-packages

WORKDIR /home/airflow/project
COPY . .

ARG USER=default
ARG PASSWORD=default
ARG EMAIL=default@default.com

RUN airflow db migrate
RUN airflow users create --username $USER \
    --firstname user \
    --lastname user \
    --role Admin \
    --email ${EMAIL} \
    --password ${PASSWORD}

RUN airflow scheduler &

EXPOSE 8080
ENTRYPOINT [ "airflow", "webserver", "--port", "8080" ]
