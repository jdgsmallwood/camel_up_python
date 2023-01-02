format:
	isort .
	black .

clean:
	rm -rf __pycache__

install:
	pip install -e .