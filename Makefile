setup:
	./setup.sh
train:
	python model.py
clean:
	rm -rf venv __pycache__