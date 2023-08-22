# CCdownscaling

This package provides implementation of several statistical climate downscaling techniques, as well as evaluation tools for downscaling outputs. 

## Requirements

See [`environment.yml`](./environment.yml). Tensorflow is pinned in this conda environment in the interest of reproducibility.

## Installation 

With conda:
```bash
git clone https://github.com/drewpolasky/CCdownscaling
cd CCdownscaling
conda env create -f environment.yml -n ccdown_env
conda activate ccdown_env 
export PYTHONPATH=$PWD:$PYTHONPATH
```
## Usage

An example use case for downscaling precipitation at Chicago O'Hare airport can be found in the example folder.
This example requires some example data, which can be downloaded from: https://zenodo.org/record/6506677

Once that data is in place, the example can be run with: 
```bash
cd example
python ohare_example.py
```
And runs through several downscaling methods, including SOM, random forest, and quantile mapping. 
All these methods are then compared on PDF skill score, KS test, RMSE, bias, and autocorrelation, 
along with the undownscaled values from the NCEP reanalysis.

There are several command line settings that can be adjusted: target variable (max_temp or precip), stationID, 
and split_type (simple, percentile, seasonal):
```bash
python ohare_example.py
```
There is also a jupyter notebook with the same example included in the example folder. 

## Citation

If you use CCdownscaling, please cite:

Andrew D. Polasky, Jenni L. Evans, Jose D. Fuentes,
CCdownscaling: A Python package for multivariable statistical climate model downscaling,
Environmental Modelling & Software,
Volume 165,
2023,
105712,
ISSN 1364-8152,
https://doi.org/10.1016/j.envsoft.2023.105712.

