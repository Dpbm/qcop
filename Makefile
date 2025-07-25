SHELL := /bin/bash
TARGET_PATH ?= "."

clean-all: clean-dataset clean-pred clean-ghz clean-model clean-checkpoints clean-history clean-gen-checkpoint

clean-dataset:
	rm -rf $(TARGET_PATH)/dataset/ $(TARGET_PATH)/dataset.csv $(TARGET_PATH)/*.h5 $(TARGET_PATH)/dataset-images.zip

clean-pred:
	rm -rf $(TARGET_PATH)/ghz-prediction.pth

clean-ghz:
	rm -rf $(TARGET_PATH)/ghz.pth $(TARGET_PATH)/ghz.jpeg

clean-model:
	rm -rf $(TARGET_PATH)/model_*

clean-checkpoints:
	rm -rf $(TARGET_PATH)/checkpoint_*

clean-gen-checkpoint:
	rm -rf $(TARGET_PATH)/gen_checkpoint.json

clean-history:
	rm -rf $(TARGET_PATH)/history.json

pack:
	zip -r $(TARGET_PATH)/dataset-images.zip $(TARGET_PATH)/dataset/

lock: 
	conda-lock -f environment.yml

clean-data-folder:
	sudo rm -rf ./data

airflow-up:
	source airflow-setup-test.sh && docker compose down && docker compose build && docker compose up -d
