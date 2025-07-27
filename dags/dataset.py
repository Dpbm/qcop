"""ETL pipeline using Airflow for dataset"""

import os

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import (
    PythonOperator,
    BranchPythonOperator,
)
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

from dataset import (
    generate_images,
    crate_dataset_folder,
    remove_duplicated_files,
    transform_images,
    start_df,
    Checkpoint,
    Stages,
)
from utils.constants import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_TARGET_FOLDER,
    DEFAULT_NEW_DIM,
    DEFAULT_MAX_TOTAL_GATES,
    DEFAULT_SHOTS,
    DEFAULT_AMOUNT_OF_CIRCUITS,
    DEFAULT_THREADS,
    images_gen_checkpoint_file,
)
from ghz import gen_circuit
from export.kaggle import upload_dataset as upload_dataset_kaggle
from export.huggingface import upload_dataset as upload_dataset_huggingface

default_args = {
    "depends_on_past": True,
}

GEN_IMAGES_TASK_ID = "gen_images"
REMOVE_DUPLICATES_TASK_ID = "remove_duplicates"
TRANSFORM_TASK_ID = "transform_images"


def next_step(checkpoint: Checkpoint) -> str:
    """
    checks teh current checkpoint and returns the next
    task_id.
    """

    print(checkpoint)

    if checkpoint.stage == Stages.GEN_IMAGES:
        return GEN_IMAGES_TASK_ID

    if checkpoint.stage == Stages.DUPLICATES:
        return REMOVE_DUPLICATES_TASK_ID

    return TRANSFORM_TASK_ID


def update_checkpoint(checkpoint: Checkpoint, stage: Stages):
    """
    Updates the checkpoint to start next task.
    """

    print(checkpoint)

    checkpoint.index = 0
    checkpoint.files = []
    checkpoint.stage = stage

    print("new")
    print(checkpoint)

    checkpoint.save()


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

    checkpoint = Checkpoint(images_gen_checkpoint_file(folder))

    branch_checkpoint = BranchPythonOperator(
        task_id="check_checkpoint", python_callable=next_step, op_args=[checkpoint]
    )
    branch_checkpoint.doc_md = """
    Choose the next task based on the current checkpoint.
    """

    gen_df = PythonOperator(
        task_id="gen_df", python_callable=start_df, op_args=[folder]
    )
    gen_df.doc_md = """
    Generate an empty dataframe and saves it as an csv file.
    """

    gen_images = PythonOperator(
        task_id=GEN_IMAGES_TASK_ID,
        python_callable=generate_images,
        op_args=[
            folder,
            DEFAULT_NUM_QUBITS,
            DEFAULT_MAX_TOTAL_GATES,
            DEFAULT_SHOTS,
            DEFAULT_AMOUNT_OF_CIRCUITS,
            DEFAULT_THREADS,
            checkpoint,
        ],
    )

    gen_images.doc_md = """
    Generate images using random circuits created using 
    Qiskit framework.
    """

    transtion_gen_to_remove = PythonOperator(
        task_id="gen_to_remove",
        python_callable=update_checkpoint,
        op_args=[checkpoint, Stages.DUPLICATES],
    )

    transtion_gen_to_remove.doc_md = """
    Update checkpoint to start removing duplicated files.
    """

    remove_duplicates = PythonOperator(
        task_id=REMOVE_DUPLICATES_TASK_ID,
        python_callable=remove_duplicated_files,
        op_args=[folder, checkpoint],
    )

    remove_duplicates.doc_md = """
    Remove files that have the same hashes.
    """

    transition_remove_to_transform = PythonOperator(
        task_id="remove_to_transform",
        python_callable=update_checkpoint,
        op_args=[checkpoint, Stages.TRANSFORM],
    )

    transition_remove_to_transform.doc_md = """
    Update checkpoint to start transforming images.
    """

    transform_img = PythonOperator(
        task_id=TRANSFORM_TASK_ID,
        python_callable=transform_images,
        op_args=[folder, DEFAULT_NEW_DIM, checkpoint],
    )

    transform_img.doc_md = """
    Get those image files and then, map them into an h5 file
    with resized and normalized images.
    """

    reset_checkpoint = PythonOperator(
        task_id="reset_checkpoint",
        python_callable=update_checkpoint,
        op_args=[checkpoint, Stages.GEN_IMAGES],
    )

    reset_checkpoint.doc_md = """
    Reset checkpoint to start again.
    """

    command = f"cd {folder} && zip -r dataset-images.zip dataset/"
    pack_img = BashOperator(task_id="pack_images", bash_command=command)

    pack_img.doc_md = """
    This task is meant to get all .jpeg images that were generated, and pack them
    inside a zip file ready to ship.
    """

    gen_ghz = PythonOperator(
        task_id="gen_ghz",
        python_callable=gen_circuit,
        op_args=[DEFAULT_NUM_QUBITS, folder, DEFAULT_NEW_DIM],
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

    send_kaggle = PythonOperator(
        task_id="send_kaggle", python_callable=upload_dataset_kaggle, op_args=[folder]
    )

    send_hf = PythonOperator(
        task_id="send_huggingface",
        python_callable=upload_dataset_huggingface,
        op_args=[folder],
    )

    send_hf.doc_md = """
    Send dataset files to huggingface
    """

    create_folder >> [gen_ghz, gen_df]
    gen_df >> branch_checkpoint

    branch_checkpoint >> gen_images
    branch_checkpoint >> remove_duplicates
    branch_checkpoint >> transform_img

    gen_images >> transtion_gen_to_remove
    transtion_gen_to_remove >> remove_duplicates
    remove_duplicates >> transition_remove_to_transform
    transition_remove_to_transform >> transform_img
    transform_img >> reset_checkpoint
    reset_checkpoint >> pack_img

    [gen_ghz, pack_img] >> trigger_dag_train
    [gen_ghz, pack_img] >> send_kaggle
    [gen_ghz, pack_img] >> send_hf
