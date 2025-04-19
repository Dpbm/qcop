clean-data:
	rm -rf dataset/ dataset.csv *.npy *.npz

gen-dataset:
	python dataset.py 

check:
	python check-images.py

train:
	python train.py
