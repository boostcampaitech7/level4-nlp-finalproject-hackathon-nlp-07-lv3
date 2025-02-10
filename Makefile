clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style
setup: set-precommit set-style-dep set-git set-dev


##### basic #####
set-git:
	git config --local commit.template .github/.gitmessage

set-style-dep:
	pip3 install ruff==0.6.8

set-precommit:
	pip3 install pre-commit==3.8.0
	pre-commit install

set-dev:
	pip3 install -r requirements.txt

set-style:
	ruff check --fix  .
	ruff format .

check-quality:
	ruff check --ignore F405,E741,F821,I001 .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
