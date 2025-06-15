"""ETL pipeline using Airflow for training"""

import os

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from train import setup_and_run_training
from args.parser import Arguments
from utils.constants import DEFAULT_TARGET_FOLDER

with DAG(dag_id="train_model", description="train vision model") as dag:
    # the env variable is meant to ease the docker image usage
    folder = os.environ.get("TARGET_FOLDER") or DEFAULT_TARGET_FOLDER
    args = Arguments()
    args.target_folder = folder

    train = PythonOperator(
        task_id="train_model", python_callable=setup_and_run_training, op_args=[args]
    )

    train.doc_md = """
    Run the training cycle
    """
