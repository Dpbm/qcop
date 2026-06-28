SHELL := /bin/bash

lock: 
	conda-lock -f environment.yml

clean-data-folder:
	sudo rm -rf ./data

run-dataset:
	python create-dataset.py --target-folder ./data --threads 20 --dataset-name-kaggle "dpbmanalysis/quantum-circuit-images" --dataset-name-hf "Dpbm/quantum-circuits"
