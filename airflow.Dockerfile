FROM python:3.12.10-slim-bullseye AS setup
WORKDIR /
COPY requirements.txt .
RUN python -m venv /proj-venv
ENV PIPENV="/proj-venv/bin/pip"
RUN ${PIPENV} install -r requirements.txt



FROM debian:bookworm-slim AS entry

RUN apt update && apt install zip make -y

WORKDIR /
COPY airflow-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh


FROM apache/airflow:slim-2.11.0-python3.12 AS serve
COPY --from=setup /proj-venv/lib/python3.12/site-packages/ /home/airflow/.local/lib/python3.12/site-packages

WORKDIR /home/airflow/project
COPY . .

WORKDIR /home/airflow/.local/bin
COPY --from=entry --chown=airflow:root /usr/bin/zip zip
COPY --from=entry --chown=airflow:root /usr/bin/make make

WORKDIR /
COPY --from=entry /entrypoint.sh .

EXPOSE 8080
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "webserver", "--port", "8080" ]
