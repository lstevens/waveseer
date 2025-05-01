.PHONY: env ingest scan cluster ui api

env:
	conda env create -f env.yml

ingest:
	wave ingest --all

scan:
	wave scan 1h

cluster:
	wave cluster --tf 1h --window 24

ui:
	wave ui

api:
	wave api --reload
