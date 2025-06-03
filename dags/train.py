"""ETL pipeline using Airflow for training"""

from datetime import timedelta

from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.providers.standard.operators.python import PythonOperator

from train import setup_and_run_training

with DAG(
    dag_id="train_model", 
    description="train vision model"
) as dag:

    wait_for_dataset_creation = ExternalTaskSensor(
        task_id="wait_dataset_creation",
        external_dag_id="build_dataset",
        external_task_id=None,  # Set to None to wait for the whole DAG
        mode="poke",
    )


    train = PythonOperator(
        task_id="train_model",
        python_callable=setup_and_run_training,
    )

    train.doc_md = """
    Run the training cycle
    """

    wait_for_dataset_creation >> train