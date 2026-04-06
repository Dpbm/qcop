"""Methods for handling circuit images."""

from itertools import product, combinations
from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.datatypes import  FilePath

Rows = List[List[Any]]

class Images:
    """Class for handling circuit images."""

    def __init__(self, folder:FilePath):
        self._folder = folder

    def generate_images(
            self,
            n_qubits:int,
            amount_circuits:int,
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
                        )
                    )

                with ThreadPoolExecutor(max_workers=total_measurement_combs) as pool:
                    threads = [pool.submit(self.generate_circuit_images, *arg) for arg in args]  # type:ignore

                    rows: Rows = []
                    for future in as_completed(threads):  # type: ignore
                        rows = [
                            *rows,
                            *[list(result.values()) for result in future.result()],
                        ]
                    callback(rows)

    @staticmethod
    def _get_combinations_of_measurements(n_qubits:int) -> List[List[int]]:
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
    def _get_bitstrings_to_int(n_qubits:int) -> List[int]:
        """Generate a list of bitstrings but converted to integers."""
        return [
            int("".join(comb), 2) for comb in product("01", repeat=n_qubits)
        ]



