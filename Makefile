clean-data:
	rm -rf dataset/ dataset.csv *.npy *.npz *.h5

gen-dataset:
	python dataset.py 

check:
	python check-images.py

train:
	python train.py
