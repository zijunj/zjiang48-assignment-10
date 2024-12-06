install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && flask run --host=0.0.0.0 --port=3000