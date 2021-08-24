# JosiePy
[![pipeline status](https://gitlab.com/rubendibattista/josiepy/badges/master/pipeline.svg)](https://gitlab.labos.polytechnique.fr/rubendibattista/josiepy/commits/master)
[![coverage report](https://gitlab.com/rubendibattista/josiepy/-/jobs/artifacts/master/raw/coverage.svg?job=badges)](https://gitlab.com/rubendibattista/josiepy/pipelines)
[![loc](https://gitlab.com/rubendibattista/josiepy/-/jobs/artifacts/master/raw/loc.svg?job=badges)](https://gitlab.com/rubendibattista/josiepy/master)
[![version](https://gitlab.com/rubendibattista/josiepy/-/jobs/artifacts/master/raw/version.svg?job=badges)](https://gitlab.com/rubendibattista/josiepy/-/releases)
![python](https://gitlab.com/rubendibattista/josiepy/-/jobs/artifacts/master/raw/python.svg?job=badges)

## A 2D PDE solver written in Python without compromising (too much) performance

### [Documentation](https://josiepy.rdb.is)


## Developer Notes
### Install
We use [`poetry`](https://python-poetry.org/docs/basic-usage/) to manage the
dependencies of the package. 

To install everything in order to be able to develop on the package

```
poetry install
```

If you want to run the [examples](./examples), you need to include the jupyter 
extra

```
poetry install -E jupyter
```

### Submodules
This repository contains `submodules`. Clone it using `--recursive`  or don't
forget to `git submodule update --init --recursive`

### Correct Jupyter Git Diffing

**`jq`**
In `.gitconfig` there's the configuration to configure
`jq` to remove useless metadata from the jupyter notebooks (in
particular the `execution_count` and the `outputs`). In order to use it
you need to include the `.gitconfig` in the `.git/config`.

    git config --local include.path ../.gitconfig

**`nbdime`** 
You need to enable the [git integration](https://nbdime.readthedocs.io/en/latest/#git-integration-quickstart)

