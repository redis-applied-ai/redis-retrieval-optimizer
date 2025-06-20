.PHONY: install format lint test clean redis-start redis-stop check-types check

install:
	poetry install --all-extras

redis-start:
	docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

redis-stop:
	docker stop redis-stack

format:
	poetry run format
	poetry run sort-imports

check-types:
	poetry run check-mypy

lint: format check-types
	
test:
	poetry run test

check: lint test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "_build" -exec rm -rf {} +
