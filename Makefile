.PHONY: install scrape run test docker-build docker-run eval

install:        ## install dependencies (incl. test tools)
	pip install -r requirements-dev.txt

scrape:         ## scrape DarGlobal + Wasalt into data/raw (takes ~10-20 min, polite rate)
	python -m scraper.run

run:            ## run the app locally on :7860
	uvicorn app.main:app --reload --port 7860

test:
	pytest -q

eval:           ## retrieval quality check on the local dataset
	python scripts/eval.py

docker-build:
	docker build -t dar-assistant .

docker-run:
	docker run --rm -p 7860:7860 --env-file .env dar-assistant
