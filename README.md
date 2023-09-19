> ⚠️ THIS REPOSITORY IS NO LONGER MAINTAINED ⚠️
>
> This was my first machine learning project, It has been so long...

# Pulsar Playground [![Documentation Status](https://readthedocs.org/projects/pulsar-playground/badge/?version=latest)](https://pulsar-playground.readthedocs.io/en/latest/?badge=latest)


Data visualization and machine learning tools for the HTRU2 dataset (R. J. Lyon, 2016).

## Online workflow
1. Explore the HTRU2 dataset [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epassaro/pulsar-playground/master?urlpath=%2Fapps%2F/ExploreDataset.ipynb)

2. Train your models in the cloud [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a7lkOa8P9LscQEbcvojIR_mMP4MtaWnv)

3. Analyse your results [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epassaro/pulsar-playground/master?urlpath=%2Fapps%2F/ModelEval.ipynb)

## Local installation
```
$ conda env create -f environment.yml
$ conda activate pulsar-playground
```
## Usage
```
$ (pulsar-playground) python train.py -h

Usage: python train.py <model_1> <model_2> ...
Available models for training: knn lgr xgb ann
```
