""" Module for common tasks. """
from os import path
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from functools import reduce

__dir__ = path.dirname(__file__)


def make_sets(filename, test_size=0.3, random_state=42, stratify=True):
    """ Splits dataset in two files: 'train.csv' and 'test.csv'. Also binarizes the labels.
    
    Parameters
    -------
    filename : str
        Input filename.
    test_size : float
        Test set ratio.
    random_state: int
        Random seed.
    stratify: bool
        Stratification by label.
    """
    data = pd.read_csv(filename)

    # Binarize labels in case they are categorical.
    lb = LabelBinarizer()
    data.iloc[:,-1] = lb.fit_transform(data.iloc[:,-1])

    if stratify:
        stratify = data.iloc[:,-1]
        
    else:
        stratify = None
        
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=stratify)

    test.to_csv( __dir__ + "/dataset/test_set.csv", index=False)
    train.to_csv( __dir__ + "/dataset/train_set.csv", index=False)


def get_n_params(model):
    """ Returns the total number of elements of a param grid.

    Parameters
    -------
    model : str
        Dictionary key from 'model' dict from models.py.
    """
    n_ = []
    for k, v in model.items():
        n_.append(len(v))
    
    n = reduce(lambda a,b: a*b, n_)

    return n


