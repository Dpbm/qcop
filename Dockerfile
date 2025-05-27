FROM apache/airflow:slim-2.11.0-python3.12


COPY . .

RUN pip install -r requirements.txt

ARG USER=default
ARG PASSWORD=default
ARG EMAIL=default@default.com

EXPOSE 8080

RUN airflow db migrate
RUN airflow users create --username $USER \
                         --firstname user \
                         --lastname user \
                         --role Admin \
                         --email ${EMAIL} \
                         --password ${PASSWORD}

ENTRYPOINT [ "airflow", "webserver" ]
