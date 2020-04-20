import subprocess

from invoke import task
from pathlib import Path


@task
def docs(c):
    DOCS_PATH = Path("./docs")
    destdir = DOCS_PATH / "build"
    sourcedir = DOCS_PATH / "source"
    SPHINX_BUILD = ["sphinx-build", sourcedir, destdir]

    subprocess.run(SPHINX_BUILD)
