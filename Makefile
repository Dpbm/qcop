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

clean-model:
	@echo "Deleting model files..."
	rm -rf ./data/checkpoint.json ./data/history.csv ./data/model_* ./data/*.pt final_model.safetensors

clean-h5:
	@echo "Deleting h5 files..."
	rm -rf ./data/*.h5

generate-images:
	[[ ! -d ./data ]] && mkdir -p ./data ; python generate_images_dataset.py --target-folder ./data --threads 100 --max-gates 100 --amount-circuits 100000  

run-embeddings:
	accelerate launch embeddings.py --target-folder ./data --batch-size 100 --preload-amount 500 --dataset-name-kaggle "dpbmanalysis/quantum-circuit-images" --dataset-name-hf "Dpbm/quantum-circuits"

run-model:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python model_dense.py --target-folder ./data --epochs 30 --load-checkpoint True --scheduler-patience 2 --batch-size 100