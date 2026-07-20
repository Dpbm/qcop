SHELL := /bin/bash

lock: 
	@echo "Locking environment..."
	conda-lock -f environment.yml

clean-data-folder:
	@echo "Removing everything from data folder..."
	rm -rf ./data

clean-embeddings:
	@echo "Deleting embeddings data..."
	rm -rf ./data/*embedding*

run-dataset:
	python create-dataset.py --target-folder ./data --threads 20 --dataset-name-kaggle "dpbmanalysis/quantum-circuit-images" --dataset-name-hf "Dpbm/quantum-circuits"

run-embeddings:
	accelerate launch embeddings.py --target-folder ./data --batch-size 100 --preload-amount 500 --dataset-name-kaggle "dpbmanalysis/quantum-circuit-images" --dataset-name-hf "Dpbm/quantum-circuits"

run-ghz:
	python ghz.py --target-folder ./data

run-model:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python model-dense.py --target-folder ./data --epochs 30 --load-checkpoint True --scheduler-patience 2 --batch-size 100


