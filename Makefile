# Makefile은 다양한 작업을 자동화해주는 데 사용됨.
# Makefile은 작업(target)과 해당 작업이 샐항할 명령어(recipe)로 구성됨.
# 각 작업은 target: dependencies 형식으로 정의되며, dependencies는 해당 작업이 실행되기 전에 실행되어야 하는 다른 작업들을 의미함

clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style
setup: set-precommit set-style-dep set-test-dep set-git set-dev
test: set-test-dep set-test


##### basic #####
set-git:
	git config --local commit.template .gitmessage

set-style-dep:
	pip3 install ruff==0.6.8

set-test-dep:
	pip3 install pytest==8.3.2

set-precommit:
	pip3 install pre-commit==3.8.0
	pre-commit install

set-dev:
	pip3 install -r requirements.txt

set-test:
	python3 -m pytest tests/

set-style:
	ruff check --fix .
	ruff format .

check-quality:
	ruff check .
	ruff format --check .

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
