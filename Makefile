clean-data:
	rm -rf dataset/ dataset.csv *.h5

gen-dataset: 
	python dataset.py 

train: del-pred
	python train.py $(CHECKPOINT)

del-pred:
	rm -rf ghz-prediction.pt

ghz: del-pred
	rm -rf ghz.jpeg
	python ghz.py

del-model:
	rm -rf model_*
