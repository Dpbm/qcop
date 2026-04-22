"""Local pipeline for dataset creation"""

import asyncio
import os

from args.parser import parse_args, Arguments
from utils.colors import Colors
from generate.dataset.dataframe import DF
from generate.dataset.images import Images, Rows
from generate.dataset.files import Files
from generate.checkpoint import Checkpoint, Stages
from ghz import gen_circuit as gen_ghz_circuit
from export import KaggleExporter, HuggingFaceExporter

def update_rows_callback(rows:Rows, checkpoint:Checkpoint, df:DF):
    df.append_rows_to_file(rows)
    checkpoint.index += 1

def transform_images_callback(checkpoint:Checkpoint):
    checkpoint.index += 1

async def main(args:Arguments):
    files_handler = Files(args.targer_folder)
    files_handler.create_dataset_folder()

    gen_ghz_circuit(args.n_qubits, args.taget_folder)
    
    checkpoint = Checkpoint.get_checkpoint(files_handler.checkpoint_path)

    df = DF(files_handler.csv_file_path)
    df.create_df_file()

    img_handler = Images(files_handler.images_path)
    
    if checkpoint.stage == Stages.GEN_IMAGES:
        img_handler.generate_images(
                args.n_qubits, 
                args.amount_qubits, 
                args.max_gates,
                args.shots,
                lambda rows: update_rows_callback(rows, checkpoint, df),
                current_index=checkpoint.index
            )
        checkpoint.next_stage()

    if checkpoint.stage == Stages.SHUFFLE:
        df_data = df.load_dataframe()
        shuffled_dataset = df.shuffle_df(df_data)
        df.save_df(shuffled_dataset)
        checkpoint.next_stage()
    
    if checkpoint.stage == Stages.DUPLICATES:
        lazy_df = df.load_lazy_frame()
        clean_df = DF.clean_duplicated_rows(lazy_df)
        duplicated_files = DF.get_duplicated_files(lazy_df, clean_df)
        df.lazy_save_to_tmp(clean_df)
        Files.remove_duplicated_files(duplicated_files)
        checkpoint.next_stage()

    if checkpoint.stage == Stages.TRANSFORM:
        img_handler.transform_images(
                files_handler.h5_file_path, 
                clean_df, 
                lambda: transform_images_callback(checkpoint)
                )
        checkpoint.next_stage()

    
    if checkpoint.stage == Stages.EXPORT:
        kaggle = KaggleExporter(args.target_folder, dataset_name=args.dataset_name)
        hf = HuggingFaceExporter(os.getenv("HUGGING_FACE_API_KEY"), args.target_folder, dataset_name=args.dataset_name)

        await asyncio.gather(
            kaggle.upload_dataset(),
            hf.upload_dataset()
        )


if __name__ == "__main__":
    try:
        args = parse_args()
        asyncio.run(main(args))
    except KeyboardInterrupt:
        exit()

