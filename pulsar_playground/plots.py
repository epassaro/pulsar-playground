""" Plotting module for data visualization and ML metrics  """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import precision_recall_curve, confusion_matrix

plt.style.use('seaborn')
scatterargs = {'size':1, 'alpha':0.4, 'legend':False, 'edgecolor':'none'}
tfs = {'none': lambda x: x*1, 'sqrt': np.sqrt, 'log': np.log}


def plot_info(data, ax=None):
    """ Summary of given dataframe.
    
    Parameters
    -------
    data : DataFrame
        Pandas dataframe.
    ax : Axes
        Matplotlib subfigure axes.
    """
    if ax is None:
        ax = plt.gca()

    text = ('Num. of examples: ' + str(data.shape[0]) +
            '\nNum. of features: ' + str(data.shape[1]-1) +
            '\nFeature names: ' + str(data.columns[0:-1].tolist()) +
            '\nTarget name: ' + '\'' + str(data.columns[-1]) + '\'' +
            '\nClasses: ' + str(data.iloc[:,-1].unique().tolist()) 
           )
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.01, 0.5, text, ha='left', va='center', size=14, linespacing=1.5)


def plot_nulls(data, ax=None):
    """ Percentage of null entries per feature (barplot).
    
    Parameters
    -------
    data : DataFrame
        Pandas dataframe.
    ax : Axes
        Matplotlib subfigure axes.
    """
    if ax is None:
        ax = plt.gca()

    nulls = data.iloc[:,:-1].isnull().sum()/data.shape[0]*100
    nulls.plot(kind='barh', ax=ax)

    ax.set_xlim(0, 100)
    ax.set_title('Percentage of null values per feature')


def plot_classprop(data, ax=None):
    """ Proportion of examples per class (pieplot).
    
    Parameters
    -------
    data : DataFrame
        Pandas dataframe.
    ax : Axes
        Matplotlib subfigure axes.  
    """
    if ax is None:
        ax = plt.gca()

    data.iloc[:,-1].value_counts(ascending=True).plot(kind='pie', autopct="%1.0f%%", ax=ax)

    ax.set_ylabel('')
    ax.set_title('Proportion of target variable')


def plot_fcorr(data, x_axis, y_axis, transform_x='none', transform_y='none', ax=None):
    """ Feature vs. feature plot (scatterplot).
    
    Parameters
    -------
    data : DataFrame
        Pandas dataframe.
    x_axis: str
        Column name from dataframe.
    y_axis: str
        Column name from dataframe.
    transform_x: str
        Dictionary key from 'tfs' dict.
    transform_y: str
        Dictionary key from 'tfs' dict.
    ax : Axes
        Matplotlib subfigure axes.
    """
    sns.scatterplot(data[x_axis].apply(tfs[transform_x]), data[y_axis].apply(tfs[transform_y]), hue=data.iloc[:,-1], ax=ax, **scatterargs)   
    
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title('Feature correlation')


def plot_hist(data, x_axis, bins=10, ax=None):
    """ Plots histograms for each class.
    
    Parameters
    -------
    data : DataFrame
        Pandas dataframe.
    x_axis: str
        Column name from dataframe.
    bins: int
        Number of bins.
    ax : Axes
        Matplotlib subfigure axes.
    """    
    if ax is None:
        ax = plt.gca()

    for c in data.iloc[:,-1].unique().tolist():
        sns.distplot(data[ data.iloc[:,-1] == c ][x_axis], bins=bins, kde=False, ax=ax)
   
    ax.set_title('Hist of ' + x_axis)
    ax.set_xlabel(x_axis)
    ax.set_ylabel('counts')


def plot_ecdf(data, x_axis, ax=None):
    """ Plots the empirical cumulative distribution for each class.
    
    Parameters
    -------
    data : DataFrame
        Pandas dataframe.
    x_axis: str
        Column name from dataframe.
    ax : Axes
        Matplotlib subfigure axes.
    """    
    if ax is None:
        ax = plt.gca()

    for c in data.iloc[:,-1].unique().tolist():
        ecdf = ECDF(data[ data.iloc[:,-1] == c][x_axis])
        sns.scatterplot(ecdf.x, ecdf.y, ax=ax, **scatterargs)

    ax.set_title('Empirical CDF')
    ax.set_xlabel(x_axis)
    ax.set_ylabel('fraction of data')


def plot_prc(y_test, y_pred_proba, threshold, ax=None):
    """ Precision and recall vs. threshold curves.
    
    Parameters
    -------
    y_test : array
        Classes from the test split.
    y_pred_proba: array
        Predicted probability.
    threshold: float
        Decision threshold.
    ax : Axes
        Matplotlib subfigure axes.
    """        
    if ax is None:
        ax = plt.gca()
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:,1])
    
    ax.set_xlim(thresholds[0] -0.02, thresholds[-1] + 0.02)
    ax.set_ylim(0.02, 1.02)
    ax.set_title('PR curve for positive class')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision & Recall')
    ax.plot(thresholds, recall[:-1], label='Recall')
    ax.plot(thresholds, precision[:-1], label='Precision')
    ax.axvline(threshold, c='black', linewidth=0.75, label='Threshold')   
    ax.legend()


def plot_cm(y_test, y_pred_proba, threshold, ax=None):
    """ Confusion matrix.
    
    Parameters
    -------
    y_test : array
        Classes from the test split.
    y_pred_proba: array
        Predicted probability.
    threshold: float
        Decision threshold.
    ax : Axes
        Matplotlib subfigure axes.
    """
    if ax is None:
        ax = plt.gca()

    y_new = np.zeros(y_pred_proba.shape[0])
    y_new[y_pred_proba[:,1] > threshold] = 1
    y_new[y_pred_proba[:,1] <= threshold] = 0
    
    cm = pd.DataFrame(confusion_matrix(y_test, y_new))
    cm.index.name = 'Actual values'
    cm.columns.name = 'Predicted values'
    
    ax.set_title('Confusion matrix')
    sns.heatmap(cm, cbar=False, cmap='Blues',\
                annot=True, fmt = 'd', linecolor='none', linewidths=1, ax=ax)


def dump_idx(y_pred_proba, threshold, filename='candidates.csv'):
    """ Save indexes of examples predicted as positive.
   
    Parameters
    -------
    y_pred_proba: array
        Predicted probability.
    threshold: float
        Decision threshold.
    filename : str
        Output file.
    """
    y_new = np.zeros(y_pred_proba.shape[0])
    y_new[y_pred_proba[:,1] > threshold] = 1
    y_new[y_pred_proba[:,1] <= threshold] = 0

    candidates = np.where(y_new == 1)
    np.savetxt(filename, np.transpose(candidates), fmt='%d')

    print('Saved to \'%s\'.' % filename)