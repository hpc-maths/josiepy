# Getting started with the examples

The `examples` folder contains a `jupyter` notebook. In order to run it you need to install few dependencies. We assume you run a fairly recent Python version (>= 3.5).

## Pre-requisite for all systems 
1. Create a directory and extract the tarball inside. 
```
tar -xvf massot-hands-on.tar.gz
```
2. You're gonna find a series of folders and files inside of it. 
```
-rw-r--r--  1 rubendibattista  staff   1.5K Jul 18 16:59 LICENSE.md
drwxr-xr-x  3 rubendibattista  staff    96B Jul 18 16:59 examples
drwxr-xr-x  7 rubendibattista  staff   224B Jul 18 16:59 josie
-rw-r--r--  1 rubendibattista  staff   229K Jul 18 17:11 massot-hands-on.tar.gz
-rw-r--r--  1 rubendibattista  staff   3.7K Jul 18 16:59 setup.py
drwxr-xr-x  8 rubendibattista  staff   256B Jul 18 16:59 tests
```

3. Your hands-on resided in `examples`. 

## Unix based
On unix systems you have few alternatives: 

### Using `virtualenv`:
1. Be sure you have `virtualenv` installed on your distribution
2. Let's create a dedicated virtualenv in your current directory
```
virtualenv cism
```

3. Let's activate it
```
source ./cism/bin/activate
```

4. Your current directory content should look like this
```
-rw-r--r--  1 rubendibattista  staff   1.5K Jul 18 16:59 LICENSE.md
drwxr-xr-x  3 rubendibattista  staff    96B Jul 18 16:59 examples
drwxr-xr-x  7 rubendibattista  staff   224B Jul 18 16:59 josie
drwxr-xr-x  6 rubendibattista  staff   192B Jul 18 17:13 cism
-rw-r--r--  1 rubendibattista  staff   229K Jul 18 17:11 massot-hands-on.tar.gz
-rw-r--r--  1 rubendibattista  staff   3.7K Jul 18 16:59 setup.py
drwxr-xr-x  8 rubendibattista  staff   256B Jul 18 16:59 tests
```

4. Staying in this directory install everything
```
pip install ".[examples]"
```

5. Go in the `examples` directory and run
```
cd examples
jupyter notebook
```

## Windows (You should evaluate to use a different OS):
1. Install the latest version of `miniconda` for Python 3: https://docs.conda.io/en/latest/miniconda.html
2. Create a conda environment in the extracted directory
```
conda create -n cism 
```
3. Activte the virtualenv
```
conda activate cism
```
4. Install all the dependencies
```
pip install ".[examples]"
```

5. Go in the `examples` directory and run
```
cd examples
jupyter notebook
```

