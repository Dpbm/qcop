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


def update_rows_callback(rows:Rows, checkpoint:Checkpoint, df:DF, inc: int):
    df.append_rows_to_file(rows)
    checkpoint.index += inc
    checkpoint.save()

def transform_images_callback(checkpoint:Checkpoint):
    checkpoint.index += 1
    checkpoint.save()

async def main(args:Arguments):
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

    if checkpoint.stage == Stages.SHUFFLE:
        print("[*] Shuffling dataset...")
        df_data = df.load_dataframe()
        shuffled_dataset = df.shuffle_df(df_data)
        df.save_df(shuffled_dataset)
        checkpoint.next_stage()
        checkpoint.save()
    
    if checkpoint.stage == Stages.DUPLICATES:
        print("[*] Removing duplicated...")
        lazy_df = df.load_lazy_frame()
        clean_df = DF.clean_duplicated_rows(lazy_df)
        duplicated_files = DF.get_duplicated_files(lazy_df, clean_df)
        df.lazy_save_to_tmp(clean_df, files_handler.df_tmp_path)
        Files.remove_duplicated_files(duplicated_files)
        Files.move_tmp_to_definitive()
        checkpoint.next_stage()
        checkpoint.save()

    if checkpoint.stage == Stages.TRANSFORM:
        print("[*] Removing duplicated (%d)..." % checkpoint.index)
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
        kaggle = KaggleExporter(args.target_folder, dataset_name=args.dataset_name)
        hf = HuggingFaceExporter(os.getenv("HUGGINGFACE_API_KEY"), args.target_folder, dataset_name=args.dataset_name)

        await asyncio.gather(
            kaggle.upload_dataset(),
            hf.upload_dataset()
        )
        checkpoint.next_stage()
        checkpoint.save()


if __name__ == "__main__":
    try:
        args = parse_args()
        asyncio.run(main(args))
    except KeyboardInterrupt:
        exit()

