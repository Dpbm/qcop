SHELL := /bin/bash

clean-all: clean-dataset clean-pred clean-ghz clean-model clean-checkpoints clean-history

clean-dataset:
	rm -rf dataset/ dataset.csv *.h5 dataset-images.zip

clean-pred:
	rm -rf ghz-prediction.pth

clean-ghz:
	rm -rf ghz.pth ghz.jpeg

clean-model:
	rm -rf model_*

clean-checkpoints:
	rm -rf checkpoint_*

clean-history:
	rm -rf history.json

pack:
	zip -r dataset-images.zip dataset/

lock: 
	conda-lock -f environment.yml

airflow-up:
	source airflow-setup-test.sh && docker compose down && docker compose build && docker compose up -d
