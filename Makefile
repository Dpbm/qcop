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

gen-dataset: 
	python dataset.py 

train: 
	python train.py --checkpoint "$(CHECKPOINT)"

pack:
	zip -r dataset-images.zip dataset/


ghz: clean-ghz
	python ghz.py

lock: 
	conda-lock -f environment.yml
