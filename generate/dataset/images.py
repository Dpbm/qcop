"""Methods for handling circuit images."""

from itertools import product, combinations
from typing import List, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torchvision.transforms import v2
from PIL import Image
import polars as pl

from utils.datatypes import  FilePath
from utils.constants import SCALE_CIRCUIT_SIZE
from .dataframe import Schema

Rows = List[List[Any]]
Measurement = List[int]
Measurements = List[Measurement]

class Images:
    """Class for handling circuit images."""

    def __init__(self, folder:FilePath):
        self._folder = folder

    def generate_images(
            self,
            n_qubits:int,
            amount_circuits:int,
            total_gates:int,
            shots:int,
            callback:Callable,
            current_index:int=0,
    ):
        """Generate the images split into threads."""

        measurement_combs = Images._get_combinations_of_measurements(n_qubits)
        total_measurement_combs = len(measurement_combs)

        with tqdm(total=amount_circuits, initial=current_index) as progress:
            while current_index < amount_circuits:
                args = []

                for i in range(total_measurement_combs):
                    img_index = current_index * total_measurement_combs + i
                    meas = measurement_combs[img_index]
                    args.append(
                        (
                            img_index,
                            meas,
                            n_qubits,
                            total_gates,
                            shots
                        )
                    )

                with ThreadPoolExecutor(max_workers=total_measurement_combs) as pool:
                    threads = [pool.submit(self._generate_circuit_images, *arg) for arg in args]

                    rows: Rows = []
                    for future in as_completed(threads):
                        rows = [
                            *rows,
                            future.result().values(),
                        ]
                        current_index += 1

                    if(len(args) == len(rows)):
                        callback(rows)

    def _generate_circuit_images(
            img_index:int, 
            meas:Measurement, 
            n_qubits:int, 
            total_gates:int,
            shots: int
    ) -> Schema:
        """ Run an experiment, save its images and return its results for different combinations 
        of measurements.
        """

        sim = AerSimulator()
        pm = generate_preset_pass_manager(backend=sim, optimization_level=0)
        sampler = Sampler()
        qc = get_random_circuit(n_qubits, total_gates)
        qc_copy = qc.copy()
        
        img_path = os.path.join(self._folder, "%d.png" % image_index)
        total_measurements = len(measurement)

        # non-interactive backend
        matplotlib.use("Agg")

        classical_register = ClassicalRegister(total_measurements, name="c")
        qc_copy.add_register(classical_register)
        qc_copy.measure(meas, list(range(total_measurements)))

        gates_count = {
                1:0, # single qubit gates
                2:0, # two qubit gates
            }

        for instruction in qc_copy.data:
            qubits = instruction.num_qubits
            if qubits not in gates_count:
                continue
            gates_count[qubits] += 1

        barriers_count = qc_copy.count_ops().get("barrier", 0)

        drawing = qc_copy.draw(
                "mpl", 
                filename=img_path, 
                fold=-1, 
                scale=SCALE_CIRCUIT_SIZE)
        plt.close(drawing)

        depth = qc_copy.depth()
        isa_qc = pm.run(qc_copy)

        with open(image_path, "rb") as file:
            img = Image.open(file)
            width, height = img.size
            file_hash = hashlib.md5(file.read()).hexdigest()
            img.close()
            
        outcomes = sampler.run([isa_qc], shots=shots).result().quasi_dists[0]
        Images._fix_dist_gaps(outcomes, n_qubits)

        return{
                "index": img_index,
                "depth": depth,
                "file": img_path,
                "result": json.dumps(list(outcomes.values())),
                "hash": file_hash,
                "total_meas": total_measurements,
                "measurements": json.dumps(measurement),
                "img_width": width,
                "img_height": height,
                "n_two_qubit_gates": gates_count.get(2, 0),
                "n_one_qubit_gates": gates_count.get(1, 0),
                "file_size_in_bytes": os.path.getsize(img_path),
                "n_barriers": barriers_count
            }

    def transform_images(
            self, 
            h5_file_path:FilePath, 
            df:pl.LazyFrame, 
            callback: Callable,
            current_index:int=0,
        ):
        """Normalize images and save them into a h5 file"""
        amount_of_rows_per_iteration = 2000

        while True:
            collected_rows: List[FilePath] = (
                df.slice(offset=current_index, length=amount_of_rows_per_iteration)
                .collect()
                .get_column("file")
                .to_list()
            )

            if len(collected_rows) <= 0:
                break
            
            # multiple threads can't write the same file, so no threads
            with h5py.File(h5_file_path, "a") as file:
                for image_path in tqdm(collected_rows):
                    with Image.open(image_path) as img:
                        tensor = Images.transform_image(img, max_width, max_height)
                        file.create_dataset(f"{image_i}", data=tensor)
                        callback()
                        current_index += 1

    @staticmethod
    def _fix_dist_gaps(dist: Dict[int,float], n_qubits:int):
        """ Set the remaining values of the distribution (remaining bitstrings) to zero """
        for i in range(2**n_qubits):
            dist[i] = dist.get(i, 0.0)

    @staticmethod
    def _get_combinations_of_measurements(n_qubits:int) -> Measurements:
        """
        Get all measurements combinations for n qubits.
        !!! It may be expensive with a many qubits  !!!
        """

        qubits_iter = list(range(n_qubits))

        measurement_combs: MeasurementsCombinations = [
            qubits_iter
        ]  # start with [[0,1,2,3,4,....,n-1]]

        for amount in range(1, n_qubits):
            measurement_combs = [
                *measurement_combs,
                *list(combinations(qubits_iter, amount)),  # type: ignore
            ]  # type: ignore

        return measurement_combs

    @staticmethod
    def _transform_image(img:Image) -> torch.Tensor:
        """
        Transform and normalize a PIL image into a torch tensor ranging values from 0 to 1.
        """

        img = img.convert("RGB")
        pipeline = [
            v2.Grayscale(),
            v2.PILToTensor(),
            v2.RandomAutocontrast(p=1),
            v2.RandomAdjustSharpness(sharpness_factor=4, p=1),
            v2.ToDtype(torch.float16, scale=True),
        ]
        return v2.Compose(pipeline)(img) / 255.0

