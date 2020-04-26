import shutil
import subprocess

from invoke import task
from pathlib import Path

from josie.nbconvert import MdBinderExporter

EXAMPLES = Path("./examples").glob("*.ipynb")
DOCS_DIR = Path("./docs")
EXAMPLES_DOCS_DIR = DOCS_DIR / "source" / "examples"
DOCS_DEPLOY_DIR = Path("./public")


@task
def copy_docs(c):
    """ Copy the Sphinx files in the build directory """

    print(f"Create `examples` dir in {DOCS_DIR}")

    EXAMPLES_DOCS_DIR.mkdir(exist_ok=True)


@task(copy_docs)
def convert_examples(c):
    """ Convert the Jupyter notebooks into RST files and move them in the
    documentation build directory"""

    print(f"Converting notebooks...")

    for filename in EXAMPLES:
        exporter = MdBinderExporter()
        output, resources = exporter.from_filename(filename)

        out_filename = (
            resources["metadata"]["name"] + resources["output_extension"]
        )

        out = EXAMPLES_DOCS_DIR / out_filename

        print(f"Converting from {filename} to {out}")

        with open(out, "w") as f:
            f.write(output)


@task(copy_docs, convert_examples)
def docs(c):
    """ This command generates the HTML of the documentation """

    dest_dir = DOCS_DIR / "build"
    source_dir = DOCS_DIR / "source"
    SPHINX_BUILD = ["sphinx-build", source_dir, dest_dir]

    subprocess.run(SPHINX_BUILD)
    shutil.copytree(DOCS_DIR / "build", DOCS_DEPLOY_DIR)
