"""ETL pipeline using Airflow for dataset"""

import os

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

from generate.dataset import (
    generate_images,
    crate_dataset_folder,
    remove_duplicated_files,
    transform_images,
    start_df,
)
from utils.constants import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_TARGET_FOLDER,
    DEFAULT_NEW_DIM,
    DEFAULT_MAX_TOTAL_GATES,
    DEFAULT_SHOTS,
    DEFAULT_DATASET_SIZE,
    DEFAULT_THREADS,
    dataset_file,
)
from generate.ghz import gen_circuit

default_args = {
    "depends_on_past": True,
}

with DAG(
    dag_id="build_dataset",
    default_args=default_args,
    description="Generate quantum circuits, map data into h5 file and upload to registries",
) as dag:
    # the env variable is meant to ease the docker image usage
    folder = os.environ.get("TARGET_FOLDER") or DEFAULT_TARGET_FOLDER
    create_folder = PythonOperator(
        task_id="create_folder",
        python_callable=crate_dataset_folder,
        op_args=[folder],
    )

    create_folder.doc_md = """
    Create a folder (if it doesn't exist) to store images.
    """

    gen_df = PythonOperator(
        task_id="gen_df", python_callable=start_df, op_args=[dataset_file(folder)]
    )
    gen_df.doc_md = """
    Generate an empty dataframe and saves it as an csv file.
    """

    gen_images = PythonOperator(
        task_id="gen_images",
        python_callable=generate_images,
        op_args=[
            folder,
            DEFAULT_NUM_QUBITS,
            DEFAULT_MAX_TOTAL_GATES,
            DEFAULT_SHOTS,
            DEFAULT_DATASET_SIZE,
            DEFAULT_THREADS,
        ],
    )

    gen_images.doc_md = """
    Generate images using random circuits created using 
    Qiskit framework.
    """

    remove_duplicates = PythonOperator(
        task_id="remove_duplicates",
        python_callable=remove_duplicated_files,
        op_args=[folder],
    )

    remove_duplicates.doc_md = """
    Remove files that have the same hashes.
    """

    transform_img = PythonOperator(
        task_id="transform_images",
        python_callable=transform_images,
        op_args=[folder, DEFAULT_NEW_DIM],
    )

    transform_img.doc_md = """
    Get those image files and then, map them into an h5 file
    with resized and normalized images.
    """

    pack_img = BashOperator(task_id="pack_images", bash_command="make pack")

    pack_img.doc_md = """
    This task is meant to get all .jpeg images that were generated, and pack them
    inside a zip file ready to ship.
    """

    gen_ghz = PythonOperator(
        task_id="gen_ghz",
        python_callable=gen_circuit,
        op_args=[DEFAULT_NUM_QUBITS, DEFAULT_TARGET_FOLDER, DEFAULT_NEW_DIM],
    )

    gen_ghz.doc_md = """
    Generate a GHZ experiment and saves the experiments results.
    """

    trigger_dag_train = TriggerDagRunOperator(
        task_id="run_training", trigger_dag_id="train_model", wait_for_completion=False
    )

    trigger_dag_train.doc_md = """
    Run training after finishing all processes.
    """

    create_folder >> [gen_ghz, gen_df]
    gen_df >> gen_images
    gen_images >> remove_duplicates
    remove_duplicates >> transform_img
    transform_img >> pack_img

    [gen_ghz, pack_img] >> trigger_dag_train
