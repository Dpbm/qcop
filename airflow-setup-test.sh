#!/bin/bash

KAGGLE_KEY_FILE="$HOME/.kaggle/kaggle.json"

clear_quotation_marks(){
    STRING=$1
    echo $STRING | sed 's/"//g'
}

export AIRFLOW_USERNAME="test"
export AIRFLOW_PASSWORD="test"
export AIRFLOW_EMAIL="test@test.com"

export ROOT_DB="rootuser"
export ROOT_PASS="rootpass"
export DB_NAME="airflow"
export DB_USERNAME="airflowuser"
export DB_PASSWORD="airflowpass"

export KAGGLE_USERNAME=$(clear_quotation_marks $(cat $KAGGLE_KEY_FILE | jq .username))
export KAGGLE_KEY=$(clear_quotation_marks $(cat $KAGGLE_KEY_FILE | jq .key))
export KAGGLE_DATASET="dpbmanalysis/quantum-circuit-images"

# HF_TOKEN comes from system
export HF_DATASET="Dpbm/quantum-circuits"

# fix permission for volume
sudo rm -rf ./data
sudo mkdir -p ./data
sudo chown -R 50000:50000 ./data

