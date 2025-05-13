clean-all: clean-dataset clean-pred clean-ghz clean-checkpoints

clean-dataset:
	rm -rf dataset/ dataset.csv *.h5 dataset-images.zip

clean-pred:
	rm -rf ghz-prediction.pt

clean-ghz:
	rm -rf ghz.pt ghz.jpeg

clean-checkpoints:
	rm -rf model_*

gen-dataset: 
	python dataset.py 

train: 
	python train.py $(CHECKPOINT)


ghz: clean-ghz
	python ghz.py

