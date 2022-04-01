.PHONY: setup install lint

setup:
	pip install --upgrade pip
	pip install "poetry==1.1.13"
	poetry config virtualenvs.create false

install: setup
	poetry install

lint:
	poetry run black rl-MEC-scheduler

.PHONY: black-test mypy-test pytest-test tests

black-test:
	poetry run black rl-MEC-scheduler --check --diff

mypy-test:
	poetry run mypy rl-MEC-scheduler

pytest-test:
	poetry run pytest --cov=rl-MEC-scheduler rl-MEC-scheduler/ --cov-report term-missing --cov-config=.coveragerc

tests: black-test mypy-test pytest-test

