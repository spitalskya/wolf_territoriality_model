visualize: FORCE
	python -m src.visualize

simulation: FORCE
	python -m src.simulation

statistics: FORCE
	python -m src.database_handler

FORCE: ;
