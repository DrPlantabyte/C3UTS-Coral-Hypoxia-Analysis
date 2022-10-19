# loading data
import numpy, pandas, os, statsmodels
from scipy import stats, integrate
from scipy.misc import derivative
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from dmba import AIC_score, BIC_score, adjusted_r2_score
from numpy import ndarray
from pandas import DataFrame
from IPython.display import display, Image, HTML
from os import path
import pyexcel as pyxl
##
this_dir = path.dirname(path.abspath(__file__))
root_dir = path.dirname(this_dir)
ex_dir = path.join(root_dir, 'Jupyter-Notebook', 'data','example')
data_dir = path.join(root_dir, 'Jupyter-Notebook', 'data','O2-curves')
example_output_imgs = list([path.join(ex_dir, f) for f in os.listdir(ex_dir) if str(f).endswith('.png')])
example_output_imgs.sort()
data_file = path.join(data_dir, 'My data sent to US for analysis.xlsx')
data_series = []
xlbook = pyxl.load_book(file_name=data_file)
i = 0
for sheet in xlbook:
    d_name = sheet.name
    plotno = (i := i+1)
    print(plotno, d_name)
    # sheet[row,col]
    columns = list([sheet[1,x] for x in range(2,7)])
    rows = []
    r=2
    while r < len(sheet.column[2]) and str(sheet[r,2]).strip() != '':
        if len(sheet.row[r]) < 7:
            continue
        rows.append(list([sheet[r,x] for x in range(2,7)]))
        r += 1
    df = DataFrame(rows, columns=columns).replace(r'^s*$', numpy.nan, regex = True)
    data_series.append((plotno, d_name, df))
print('loaded %s data sets' % len(data_series))