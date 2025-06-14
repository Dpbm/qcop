#!/bin/bash

until pg_isready -h db; do
	echo "Not ready yet..."
	sleep 2
done

echo "Migrating...."
airflow db migrate

# USER, PASSWORD and EMAIl come from env variables
echo "Creating Airflow user ${USER}..."
airflow users create \
    --username $USER \
    --firstname user \
    --lastname user \
    --role Admin \
    --email ${EMAIL} \
    --password ${PASSWORD}

echo "Running scheduler...."
airflow scheduler &

exec airflow "$@"
