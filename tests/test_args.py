import sys
import pytest
from args.parser import parse_args
from utils.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT,
    DEFAULT_AMOUNT_OF_CIRCUITS,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_TOTAL_GATES,
    DEFAULT_NEW_DIM,
    DEFAULT_NUM_QUBITS,
    DEFAULT_SHOTS,
    DEFAULT_TARGET_FOLDER,
    DEFAULT_TEST_PERCENTAGE,
    DEFAULT_THREADS,
    DEFAULT_TRAIN_PERCENTAGE,
)


@pytest.fixture(autouse=True)
def clear_argv():
    """
    Clear command line arguments to avoid errors.
    """
    first_argument = sys.argv[0]
    sys.argv.clear()
    sys.argv.append(first_argument)


class TestArgs:
    """
    Test the parsing of command line arguments.
    """

    def test_default_values(self):
        """Check if the default parameters are ok"""

        args = parse_args()

        assert args.batch_size == DEFAULT_BATCH_SIZE
        assert args.checkpoint == DEFAULT_CHECKPOINT
        assert args.amount_circuits == DEFAULT_AMOUNT_OF_CIRCUITS
        assert args.epochs == DEFAULT_EPOCHS
        assert args.max_gates == DEFAULT_MAX_TOTAL_GATES
        assert args.n_qubits == DEFAULT_NUM_QUBITS
        assert args.new_image_dim == DEFAULT_NEW_DIM
        assert args.shots == DEFAULT_SHOTS
        assert args.target_folder == DEFAULT_TARGET_FOLDER
        assert args.test_size == DEFAULT_TEST_PERCENTAGE
        assert args.threads == DEFAULT_THREADS
        assert args.train_size == DEFAULT_TRAIN_PERCENTAGE

    def test_arbitrary_arguments(self):
        sys.argv = [
            *sys.argv,
            "--epochs",
            "10",
            "--batch-size",
            "3",
            "--train-size",
            "0.3",
            "--test-size",
            "0.5",
            "--checkpoint",
            "test",
            "--threads",
            "5",
            "--shots",
            "100000",
            "--n-qubits",
            "6",
            "--max-gates",
            "103",
            "--amount-circuits",
            "3233",
            "--target-folder",
            "another",
            "--new-image-dim",
            "83",
            "123",
        ]

        args = parse_args()

        assert args.epochs == int(sys.argv[2])
        assert args.batch_size == int(sys.argv[4])
        assert args.train_size == float(sys.argv[6])
        assert args.test_size == float(sys.argv[8])
        assert args.checkpoint == sys.argv[10]
        assert args.threads == int(sys.argv[12])
        assert args.shots == int(sys.argv[14])
        assert args.n_qubits == int(sys.argv[16])
        assert args.max_gates == int(sys.argv[18])
        assert args.amount_circuits == int(sys.argv[20])
        assert args.target_folder == sys.argv[22]
        assert args.new_image_dim == [int(sys.argv[24]), int(sys.argv[25])]
