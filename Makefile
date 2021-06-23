help:
	@echo "Commands:"
	@echo ""
	@echo "  install        install in editable mode"
	@echo "  dev-install    install in editable mode with dev requirements"
	@echo "  pytest         run the test suite and report coverage"
	@echo "  flake8         style check with flake8"
	@echo "  html           build docs (update existing)"
	@echo "  html-noplot    as above, without gallery"
	@echo "  html-clean     build docs (new, removing any existing)"
	@echo "  preview        renders docs in Browser"
	@echo "  linkcheck      check all links in docs"
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
	pytest --cov=empymod --flake8 --mpl
	coverage html
	rm  matplotlibrc

flake8:
	flake8 docs/conf.py setup.py empymod/ tests/ examples/

html:
	cd docs && make html

html-noplot:
	cd docs && make html-noplot

html-clean:
	cd docs && rm -rf api/empymod* gallery/*/ _build/ && make html

preview:
	xdg-open docs/_build/html/index.html

linkcheck:
	cd docs && make linkcheck

clean:
	rm -rf build/ dist/ .eggs/ empymod.egg-info/ empymod/version.py  # build
	rm -rf */__pycache__/ */*/__pycache__/      # python cache
	rm -rf .coverage htmlcov/ .pytest_cache/    # tests and coverage
	rm -rf docs/gallery/*/ docs/gallery/*.zip docs/_build/ docs/api/empymod* # docs
	rm -rf matplotlibrc docs/savefig
	rm -rf filters/ examples/educational/filters/
