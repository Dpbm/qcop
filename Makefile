clean-data:
	rm -rf dataset/ dataset.csv *.npy *.npz *.h5

gen-dataset:
	python dataset.py 

check:
	python check-images.py

train: del-pred
	python train.py

del-pred:
	rm -rf ghz-prediction.pt
