"""ETL pipeline using Airflow for training"""

import os

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from train import setup_and_run_training
from args.parser import Arguments
from utils.constants import DEFAULT_TARGET_FOLDER
from utils.helpers import get_latest_model_checkpoint
from export.kaggle import upload_model as upload_model_kaggle
from export.huggingface import upload_model as upload_model_hf

with DAG(dag_id="train_model", description="train vision model") as dag:
    # the env variable is meant to ease the docker image usage
    folder = os.environ.get("TARGET_FOLDER") or DEFAULT_TARGET_FOLDER
    args = Arguments()
    args.target_folder = folder

    checkpoint = get_latest_model_checkpoint(folder)
    if(checkpoint):
        args.checkpoint = checkpoint

    train = PythonOperator(
        task_id="train_model", 
        python_callable=setup_and_run_training, 
        op_args=[args]
    )

    train.doc_md = """
    Run the training cycle
    """

    upload_kaggle = PythonOperator(
        task_id="upload_kaggle",
        python_callable=upload_model_kaggle,
        op_args=[folder]
    )

    upload_kaggle.doc_md = """
    Send model file to kaggle
    """
    
    upload_hf = PythonOperator(
        task_id="upload_hugginface",
        python_callable=upload_model_hf,
        op_args=[folder]
    )

    upload_hf.doc_md = """
    Send model file to huggingface
    """

    train >> upload_kaggle
    train >> upload_hf
