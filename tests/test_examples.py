import os
import pytest
import subprocess
import warnings

from pathlib import Path

from josie.nbconvert import NoMagicPythonExporter

EXAMPLES_PATH = Path(__file__).parents[1] / "examples"
notebooks = EXAMPLES_PATH.glob("*.ipynb")


@pytest.fixture(scope="module", params=notebooks)
def example(request):
    notebook = request.param

    base_dir = notebook.parent

    # Convert all the examples from jupyter notebook to python files
    exporter = NoMagicPythonExporter()

    # Ignore this warning coming from nbconvert
    # TODO: In the future they will probably fix it upstream
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        output, resources = exporter.from_filename(notebook)

    # Get the converted notebooks
    out_filename = (
        resources["metadata"]["name"] + resources["output_extension"]
    )

    py_notebook = base_dir / out_filename

    print(f"Converting from {notebook} to {py_notebook}")

    with open(py_notebook, "w") as f:
        f.write(output)

    yield py_notebook

    # Remove the converted notebook
    os.remove(py_notebook)


@pytest.mark.bench
def test_example(example, benchmark):
    @benchmark
    def run_example():
        subprocess.run(["python", f"{example}"])
