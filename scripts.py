import subprocess
import sys


def format():
    subprocess.run(["isort", "./redis_retrieval_optimizer", "./tests/", "--profile", "black"], check=True)
    subprocess.run(["black", "./redis_retrieval_optimizer", "./tests/"], check=True)


def check_format():
    subprocess.run(["black", "--check", "./redis_retrieval_optimizer"], check=True)


def sort_imports():
    subprocess.run(["isort", "./redis_retrieval_optimizer", "./tests/", "--profile", "black"], check=True)


def check_sort_imports():
    subprocess.run(
        ["isort", "./redis_retrieval_optimizer", "--check-only", "--profile", "black"], check=True
    )


def check_lint():
    subprocess.run(["pylint", "--rcfile=.pylintrc", "./redis_retrieval_optimizer"], check=True)


def check_mypy():
    subprocess.run(["python", "-m", "mypy", "./redis_retrieval_optimizer"], check=True)


def test():
    test_cmd = ["python", "-m", "pytest"]
    # Get any extra arguments passed to the script
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)
    subprocess.run(test_cmd, check=True)
