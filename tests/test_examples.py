import glob
import os
import pytest
import shlex
import subprocess

EXAMPLES_PATH = os.path.abspath("./examples")
EXAMPLES_CONFIG_PATH = os.path.join(EXAMPLES_PATH, "config")
CONF_FILE = os.path.join(EXAMPLES_CONFIG_PATH, "nbconvert_config.py")
TEMPLATE_FILE = os.path.join(EXAMPLES_CONFIG_PATH, "python_nomagic.tpl")
notebooks = glob.glob(os.path.join(EXAMPLES_PATH, "*.ipynb"))


@pytest.fixture(scope="module", params=notebooks)
def example(request):
    notebook = request.param
    # Convert all the examples from jupyter notebook to python files
    cmd = [
        "jupyter",
        "nbconvert",
        "--config",
        f"{shlex.quote(CONF_FILE)}",
        "--to",
        "python",
        "--template",
        f"{shlex.quote(TEMPLATE_FILE)}",
        f"{notebook}",
    ]

    # Convert
    subprocess.run(cmd)

    # Get the converted notebooks
    py_notebook = notebook.replace(".ipynb", ".py")

    yield py_notebook

    # Remove the converted notebook
    os.remove(py_notebook)


@pytest.mark.bench
def test_example(example, benchmark):
    @benchmark
    def run_example():
        subprocess.run(["python", f"{example}"])
