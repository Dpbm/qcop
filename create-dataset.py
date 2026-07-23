"""Local pipeline for dataset creation"""

import argparse
import asyncio
import os

from utils.colors import Colors
from generate.dataset.dataframe import DF
from generate.dataset.images import Images, Rows
from generate.dataset.files import Files
from generate.checkpoint import Checkpoint, Stages
from ghz import gen_circuit as gen_ghz_circuit
from export import KaggleExporter, HuggingFaceExporter, export_parallel
from utils.constants import (
    DEFAULT_SHOTS,
    DEFAULT_NUM_QUBITS,
    DEFAULT_MAX_TOTAL_GATES,
    DEFAULT_THREADS,
    DEFAULT_AMOUNT_OF_CIRCUITS,
    DEFAULT_DATASET_NAME
)

def update_rows_callback(rows:Rows, checkpoint:Checkpoint, df:DF, inc: int):
    df.append_rows_to_file(rows)
    checkpoint.index += inc
    checkpoint.save()

def transform_images_callback(checkpoint:Checkpoint):
    checkpoint.index += 1
    checkpoint.save()

async def main(args:argparse.Namespace):
    print("[*] Setting up files...")
    files_handler = Files(args.target_folder)
    files_handler.create_dataset_folder()
    
    print("[*] Creating GHZ...")
    gen_ghz_circuit(args.n_qubits, args.target_folder)
    
    checkpoint = Checkpoint.get_checkpoint(files_handler.checkpoint_path)

    print("[*] Creating DF CSV File...")
    df = DF(files_handler.csv_file_path)
    df.create_df_file()

    img_handler = Images(files_handler.images_path)
    
    if checkpoint.stage == Stages.GEN_IMAGES:
        print("[*] Generating images (%d)..."%checkpoint.index)
        img_handler.generate_images(
                args.n_qubits, 
                args.amount_circuits, 
                args.max_gates,
                args.shots,
                lambda rows,inc: update_rows_callback(rows, checkpoint, df, inc),
                args.threads,
                checkpoint,
                current_index=checkpoint.index
            )
        checkpoint.next_stage()
        checkpoint.save()
    
    if checkpoint.stage == Stages.CLEAN:
        print("[*] Cleaning Data...")
        DF.run_statistics_notebook(files_handler.pre_analysis_path)

        lazy_df = df.load_lazy_frame()
        clean_df = DF.clean_duplicated_rows(lazy_df)

        duplicated_files = DF.get_files_via_left_join(lazy_df, clean_df)
        dont_exist = Files.remove_duplicated_files(duplicated_files)
        clean_df = DF.remove_rows_with_non_existant_files(clean_df, dont_exist)

        clean_df = DF.keep_25_to_75_quantiles_by_depth(clean_df)
        discard_outliers_files = DF.get_files_via_left_join(lazy_df, clean_df)
        
        dont_exist = Files.remove_duplicated_files(duplicated_files)
        clean_df = DF.remove_rows_with_non_existant_files(clean_df, dont_exist)

        df.lazy_save_to_tmp(clean_df, files_handler.df_tmp_path)
        files_handler.move_tmp_to_definitive()
        
        DF.run_statistics_notebook(files_handler.post_analysis_path)

        checkpoint.next_stage()
        checkpoint.save()

    if checkpoint.stage == Stages.TRANSFORM:
        print("[*] Transforming images (%d)..." % checkpoint.index)
        clean_df = df.load_lazy_frame()
        img_handler.transform_images(
                files_handler.h5_file_path, 
                clean_df, 
                lambda: transform_images_callback(checkpoint),
                current_index=checkpoint.index,
                )
        checkpoint.next_stage()
        checkpoint.save()

    
    if checkpoint.stage == Stages.EXPORT:
        print("[*] Exporting...")
        await export_parallel(args.target_folder, args.dataset_name_kaggle,os.getenv("HUGGINGFACE_API_KEY"), args.dataset_name_hf)
        checkpoint.next_stage()
        checkpoint.save()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset-name-kaggle", type=str, default=DEFAULT_DATASET_NAME)
        parser.add_argument("--dataset-name-hf", type=str, default=DEFAULT_DATASET_NAME)
        parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)

        parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
        parser.add_argument("--n-qubits", type=int, default=DEFAULT_NUM_QUBITS)
        parser.add_argument("--max-gates", type=int, default=DEFAULT_MAX_TOTAL_GATES)

        parser.add_argument(
            "--amount-circuits", type=int, default=DEFAULT_AMOUNT_OF_CIRCUITS
        )
        parser.add_argument("--target-folder", type=str, required=True)

    if len(sys.argv) <= 2:
        parser.print_usage()
        exit()
        
        args = parser.parse_args()
        parsed_arguments = ArgumentsGenerate()
        parsed_arguments.parse(args)

        asyncio.run(main(args))
    except KeyboardInterrupt:
        exit()

