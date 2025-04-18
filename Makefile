clean-data:
	rm -rf dataset/ dataset.csv

del-duplicated:
	python clean-images.py

gen-dataset:
	python dataset.py 
