import pytest

import nbformat

from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor

EXAMPLES_PATH = Path(__file__).parents[1] / "examples"
notebooks = tuple(EXAMPLES_PATH.glob("*.ipynb"))
ids = [path.name for path in notebooks]


@pytest.mark.bench
@pytest.mark.parametrize("notebook", notebooks, ids=ids)
def test_example(notebook, benchmark):
    with open(notebook, "r") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=None)

    benchmark(ep.preprocess, nb, {"metadata": {"path": EXAMPLES_PATH}})
