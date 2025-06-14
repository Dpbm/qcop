#!/bin/bash

echo "Waiting for PostgreSQL to start..."
until pg_isready -U $POSTGRES_USER; do
	echo "Not ready yet..."
	sleep 2
done

echo "Creating new db ${DB_NAME}..."
psql -U $POSTGRES_USER -c "CREATE DATABASE ${DB_NAME};"

echo "Creating new user ${DB_USERNAME}..."
psql -U $POSTGRES_USER -c "CREATE USER ${DB_USERNAME} WITH PASSWORD '${DB_PASSWORD}';"
psql -U $POSTGRES_USER -c "ALTER USER ${DB_USERNAME} SET search_path = public;"

echo "Setting permissions..."
psql -U $POSTGRES_USER -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USERNAME};"
psql -U $POSTGRES_USER -c "ALTER DATABASE ${DB_NAME} OWNER TO ${DB_USERNAME};"
psql -U $POSTGRES_USER -d $DB_NAME -c "GRANT ALL ON SCHEMA public TO ${DB_USERNAME};"

echo "host $DB_NAME $DB_USERNAME all trust" >> /var/lib/postgresql/data/pg_hba.conf

pg_ctl reload
