"""ETL pipeline using Airflow"""

from datetime import timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


default_args = {
    "depends_on_past": True,
    "retries": 3,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="build_dataset", 
    default_args=default_args,
    description="Generate quantum circuits, map data into h5 file and upload to registries"
) as dag:

    pack_task = BashOperator(
        task_id="pack_images",
        bash_command="make pack"
    )

    pack_task.doc_md = """
    This task is meant to get all .jpeg images that were generated, and pack them
    inside a zip file ready to ship.
    """


    pack_task
