# JosiePy
[![pipeline status](https://gitlab.com/rubendibattista/josiepy/badges/master/pipeline.svg)](https://gitlab.labos.polytechnique.fr/rubendibattista/josiepy/commits/master)
[![coverage report](https://gitlab.com/rubendibattista/josiepy/-/jobs/artifacts/master/raw/coverage.svg?job=badges)](https://gitlab.com/rubendibattista/josiepy/pipelines)
[![loc](https://gitlab.com/rubendibattista/josiepy/-/jobs/artifacts/master/raw/loc.svg?job=badges)](https://gitlab.com/rubendibattista/josiepy/master)
[![version](https://gitlab.com/rubendibattista/josiepy/-/jobs/artifacts/master/raw/version.svg?job=badges)](https://gitlab.com/rubendibattista/josiepy/-/releases)

## A 2D PDE solver written in Python without compromising (too much) performance

### [Documentation](https://josiepy.rdb.is)

### Install 
Since for the moment there are some [vendored](vendor) libraries, the library
must be installed using `pipenv`. 

```
pipenv install
```

## Developer Notes
### Install

```
pipenv install --dev
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

