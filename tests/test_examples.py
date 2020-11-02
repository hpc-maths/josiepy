import pytest

import nbformat

from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor

EXAMPLES_PATH = Path(__file__).parents[1] / "examples"
notebooks = tuple(EXAMPLES_PATH.glob("*.ipynb"))
ids = [path.name for path in notebooks]


@pytest.fixture(params=notebooks, ids=ids)
def notebook(request):
    with open(request.param, "r") as f:
        nb = nbformat.read(f, as_version=4)

    yield nb


def test_examples(notebook):
    ep = ExecutePreprocessor()
    ep.preprocess(notebook, resources={"metadata": {"path": EXAMPLES_PATH}})


@pytest.mark.bench
def bench_examples(notebook, benchmark):
    ep = ExecutePreprocessor()

    benchmark(ep.preprocess, notebook, {"metadata": {"path": EXAMPLES_PATH}})
