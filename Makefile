help:
	@echo "Commands:"
	@echo ""
	@echo "  install        install in editable mode"
	@echo "  dev-install    install in editable mode with dev requirements"
	@echo "  pytest         run the test suite and report coverage"
	@echo "  flake8         style check with flake8"
	@echo "  clean          clean up all generated files"
	@echo ""

install:
	pip install -e .

dev-install:
	pip install -r requirements-dev.txt && pip install -e .

.ONESHELL:
pytest:
	cp tests/matplotlibrc .
	rm -rf .coverage htmlcov/ .pytest_cache/
	pytest --cov=empymod --mpl
	coverage html
	rm  matplotlibrc

flake8:
	flake8 docs/conf.py setup.py empymod/ tests/ examples/

clean:
	pip uninstall empymod_plain -y
	rm -rf build/ dist/ .eggs/ empymod_plain.egg-info/  # build
	rm -rf */__pycache__/ */*/__pycache__/      # python cache
	rm -rf .coverage htmlcov/ .pytest_cache/    # tests and coverage
	rm -rf matplotlibrc docs/savefig
