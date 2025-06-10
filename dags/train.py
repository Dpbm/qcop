"""ETL pipeline using Airflow for training"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from train import setup_and_run_training

with DAG(
    dag_id="train_model", 
    description="train vision model"
) as dag:

    train = PythonOperator(
        task_id="train_model",
        python_callable=setup_and_run_training,
    )

    train.doc_md = """
    Run the training cycle
    """
