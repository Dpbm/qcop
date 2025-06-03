#!/bin/bash

export AIRFLOW_USERNAME="test"
export AIRFLOW_PASSWORD="test"
export AIRFLOW_EMAIL="test@test.com"
export ROOT_DB="rootuser"
export ROOT_PASS="rootpass"
export DB_NAME="airflow"
export DB_USERNAME="airflowuser"
export DB_PASSWORD="airflowpass"

# fix permission for volume
sudo rm -rf ./data
sudo mkdir -p ./data
sudo chown -R 50000:50000 ./data

