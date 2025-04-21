clean-data:
	rm -rf dataset/ dataset.csv *.npy *.npz *.h5

gen-dataset:
	python dataset.py 

train: del-pred
	python train.py

del-pred:
	rm -rf ghz-prediction.pt

ghz: del-pred
	rm -rf ghz.jpeg
	python ghz.py
	
