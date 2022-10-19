import os, re
from os import path
import pandas
from pandas import DataFrame
import numpy
from typing import List
from numpy import ndarray
from sklearn.base import TransformerMixin

class SelectColumns(TransformerMixin):
	def __init__(self, columns: list):
		self.columns = columns

	def fit(self, X: ndarray, y=None):
		pass

	def transform(self, X: DataFrame):
		return X[self.columns]