import os, re
from os import path
import pandas
from pandas import DataFrame
import numpy
from typing import List
from numpy import ndarray


def scan_dir(dir: str, regex='.*') -> List[str]:
	out = []
	for root, dirs, files in os.walk(str(dir)):
		for f in files:
			fpath = str(path.join(root, f)).replace(os.sep, '/')
			if re.match(regex, fpath) is not None:
				out.append(fpath)
	return out

def parse_hypoxia_csv(filepath, header_row=1,) -> DataFrame:
	row_skip = header_row
	if row_skip < 0: row_skip = None
	df= pandas.read_csv(filepath, skiprows=row_skip)
	return df[['Time (s)', 'Time (m)', 'PO2 (%)', 'O2 (mg/L)', 'VO2 (mg O2/hr)']]


def _as_df(X, rows_first=True, columns=None) -> DataFrame:
	if type(X) is DataFrame:
		return X
	if type(X) is ndarray:
		dims = X.shape
		if len(dims) == 0:
			X = numpy.asarray([[X]])
		elif len(dims) == 1:
			X = numpy.asarray([X])
		elif len(dims) > 2:
			raise TypeError(
				"Cannot convert %s-dimensional array with shape %s to DataFrame" % (len(dims), dims))
		if not rows_first:
			X = X.T
		return DataFrame(X, columns=columns)
	if type(X) is dict:
		return DataFrame.from_dict(X)
	if rows_first:
		return DataFrame(X, columns=columns)
	else:
		return DataFrame(zip(*X), columns=columns)