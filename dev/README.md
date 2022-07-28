# development folder

This folder contains scripts used to prepare this tutorial.

## extracting data form original ROOT-file source

For ease of use, we want to extract the data out of the ROOT format.
We will need the [larcv](https://github.com/larbys/larcv), which can read the original format.
We save into numpy compressed binary files (.npz), sorting the data into folders named by the class.
This will allow us to use `torchvision.datasets.DatasetFolder`.

We assume you have ROOT 6 installed. In this folder start by:

```
git clone https://github.com/larbys/larcv
cd larcv
mkdir build
alias python=python3
source configure.sh
cd build
cmake -DUSE_PYTHON3=ON ../
make install
```


