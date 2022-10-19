from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Tuple
import numpy, pandas
from numpy import ndarray, float64
from numpy.ma import masked_array
from pandas import DataFrame, Series
from scipy.optimize import curve_fit, minimize, OptimizeResult
from dmba import AIC_score
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
from statsmodels.tools.eval_measures import rmse


class RespirationModelABC(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
	@abstractmethod
	def name(self) -> str:
		pass

	@abstractmethod
	def AIC_score(self, X: ndarray, y: ndarray) -> float:
		pass

	@abstractmethod
	def fit(self, X: ndarray, y: ndarray):
		pass

	@abstractmethod
	def predict(self, X: ndarray) -> ndarray:
		pass

	@abstractmethod
	def dydx(self, X: ndarray) -> ndarray:
		pass

	def fit_predict(self, X, y=None) -> ndarray:
		self.fit(X, y)
		return self.predict(X)

class BestPoly(RespirationModelABC):
	def __init__(self, max_order):
		self.max_order = max_order
		self.fit_params = numpy.asarray([0])

	def name(self) -> str:
		return "Best Poly"

	def AIC_score(self, X: ndarray, y: ndarray):
		X, y = _check_inputs(X, y)
		y_pred = self.predict(X)
		return AIC_score(y.ravel(), y_pred.ravel(), df=len(self.fit_params))

	def predict(self, X: ndarray) -> ndarray:
		X, _ = _check_inputs(X, None)
		return polynomial0(X, self.fit_params)

	def fit(self, X: ndarray, y: ndarray):
		X, y = _check_inputs(X, y)
		best_AIC = numpy.inf
		p0 = numpy.asarray([], dtype=float64)
		best_params = p0
		for order in range(1, self.max_order+1):
			p0 = numpy.append(p0, 0.)
			p_optimum, p_covariance = curve_fit(f=polynomial0, xdata=X[:,0], ydata=y[:,0], p0=p0)
			self.fit_params = p_optimum
			this_AIC = self.AIC_score(X, y)
			if this_AIC < best_AIC:
				best_AIC = this_AIC
				best_params = p_optimum
		self.fit_params = best_params
		return self

	def dydx(self, X: ndarray) -> ndarray:
		## Note: params doesn't include the Y-offset (which is zero), so it has to be added before deriving
		return numpy.polyval(numpy.polyder(numpy.append(self.fit_params, 0.)), X)

def polynomial0(xdata, *params):
	## polynomial fixed to go through the origin
	p0 = numpy.append(params,0)
	return numpy.polyval(p0, xdata)


class ModelS1(RespirationModelABC):
	def __init__(self):
		pass # no params

	def name(self) -> str:
		return "S1"

	def AIC_score(self, X: ndarray, y: ndarray):
		X, y = _check_inputs(X, y)
		y_pred = self.predict(X)
		return AIC_score(y.ravel(), y_pred.ravel(), df=0)

	def predict(self, X: ndarray) -> ndarray:
		X, _ = _check_inputs(X, None)
		return X

	def fit(self, X: ndarray, y: ndarray):
		# do nothing
		return self

	def dydx(self, X: ndarray) -> ndarray:
		## Note: params doesn't include the Y-offset (which is zero), so it has to be added before deriving
		return numpy.zeros_like(X)

class ModelS2(RespirationModelABC):
	def __init__(self):
		self.Pt = 0.5

	def name(self) -> str:
		return "S2"

	def AIC_score(self, X: ndarray, y: ndarray):
		X, y = _check_inputs(X, y)
		y_pred = self.predict(X)
		return AIC_score(y.ravel(), y_pred.ravel(), df=1)

	def predict(self, X: ndarray) -> ndarray:
		X, _ = _check_inputs(X, None)
		# y1 = masked_array(y, mask=(X <= self.Pt) )
		mask1 = masked_array(numpy.ones_like(X), mask=(X < self.Pt)).filled(0.)
		mask2 = masked_array(numpy.ones_like(X), mask=(X >= self.Pt)).filled(0.)
		y = (mask1 * (X / self.Pt)) + (mask2 * (numpy.ones_like(X)))
		return y

	def fit(self, X: ndarray, y: ndarray):
		def cost(params, *args):
			self.Pt = params[0]
			return rmse(y.ravel(), self.predict(X).ravel())
		p0 = numpy.asarray([self.Pt])
		opt: OptimizeResult = minimize(fun=cost, x0=p0)
		self.Pt = opt['x'][0]
		return self

	def dydx(self, X: ndarray) -> ndarray:
		## Note: params doesn't include the Y-offset (which is zero), so it has to be added before deriving
		X, _ = _check_inputs(X, None)
		# y1 = masked_array(y, mask=(X <= self.Pt) )
		mask1 = masked_array(numpy.ones_like(X), mask=(X < self.Pt)).filled(0.)
		mask2 = masked_array(numpy.ones_like(X), mask=(X >= self.Pt)).filled(0.)
		dy = (mask1 * (1.0 / self.Pt)) + (mask2 * (0.))
		return dy


class ModelS3(RespirationModelABC):
	def __init__(self):
		self.Pt = 0.5
		self.Vt = 0.5

	def name(self) -> str:
		return "S3"

	def AIC_score(self, X: ndarray, y: ndarray):
		X, y = _check_inputs(X, y)
		y_pred = self.predict(X)
		return AIC_score(y.ravel(), y_pred.ravel(), df=2)

	def predict(self, X: ndarray) -> ndarray:
		X, _ = _check_inputs(X, None)
		mask1 = masked_array(numpy.ones_like(X), mask=(X < self.Pt)).filled(0.)
		mask2 = masked_array(numpy.ones_like(X), mask=(X >= self.Pt)).filled(0.)
		y = (mask1 * (X * self.Vt / self.Pt)) + (mask2 * (self.Vt + (X - self.Pt) * (1 - self.Vt) / (1 - self.Pt)))
		return y

	def fit(self, X: ndarray, y: ndarray):
		def cost(params, *args):
			self.Pt = params[0]
			self.Vt = params[1]
			return rmse(y.ravel(), self.predict(X).ravel())
		p0 = numpy.asarray([self.Pt, self.Vt])
		opt: OptimizeResult = minimize(fun=cost, x0=p0)
		self.Pt = opt['x'][0]
		self.Vt = opt['x'][1]
		return self

	def dydx(self, X: ndarray) -> ndarray:
		## Note: params doesn't include the Y-offset (which is zero), so it has to be added before deriving
		X, _ = _check_inputs(X, None)
		# y1 = masked_array(y, mask=(X <= self.Pt) )
		mask1 = masked_array(numpy.ones_like(X), mask=(X < self.Pt)).filled(0.)
		mask2 = masked_array(numpy.ones_like(X), mask=(X >= self.Pt)).filled(0.)
		dy = (mask1 * (self.Vt / self.Pt)) + (mask2 * ((1 - self.Vt) / (1 - self.Pt)))
		return dy



class MichaelisMenten(RespirationModelABC):
	def __init__(self):
		self.Vmax = 1.0
		self.Km = 0.5

	def name(self) -> str:
		return "SK"

	def AIC_score(self, X: ndarray, y: ndarray):
		X, y = _check_inputs(X, y)
		y_pred = self.predict(X)
		return AIC_score(y.ravel(), y_pred.ravel(), df=2)

	def predict(self, X: ndarray) -> ndarray:
		X, _ = _check_inputs(X, None)
		y = (self.Vmax * X) / (self.Km + X)
		return y

	def fit(self, X: ndarray, y: ndarray):
		def cost(params, *args):
			self.Vmax = params[0]
			self.Km = params[1]
			return rmse(y.ravel(), self.predict(X).ravel())
		p0 = numpy.asarray([self.Vmax, self.Km])
		opt: OptimizeResult = minimize(fun=cost, x0=p0)
		self.Vmax = opt['x'][0]
		self.Km = opt['x'][1]
		return self

	def dydx(self, X: ndarray) -> ndarray:
		## Note: params doesn't include the Y-offset (which is zero), so it has to be added before deriving
		X, _ = _check_inputs(X, None)
		# y1 = masked_array(y, mask=(X <= self.Pt) )
		dy = self.Vmax * self.Km / numpy.square(self.Km + X)
		return dy


class FixedMinMaxScaler(TransformerMixin, BaseEstimator):
	def __init__(self, min_values: ndarray, max_values: ndarray, dtype=numpy.float64):
		self.min_values = numpy.asarray(min_values, dtype=dtype)
		self.max_values = numpy.asarray(max_values, dtype=dtype)
		if self.min_values.shape != self.max_values.shape or len(self.min_values.shape) != 1:
			raise ValueError('min_values and max_values must be 1D arrays of equal length!')

	def fit(self, X, y=None, **fit_params):
		return self

	def transform(self, X, **fit_params):
		X, _ = _check_inputs(X, None)
		rng = self.max_values - self.min_values
		return (X - self.min_values) / rng

	def inverse_transform(self, X):
		rng = self.max_values - self.min_values
		return (X * rng) + self.min_values



def _check_inputs(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
	if X is not None:
		X = numpy.asarray(X).astype(numpy.float64)
		if len(X.shape) < 2:
			## must be 2D with order of [row, column], EVEN FOR 1D DATA!
			X = X.reshape((-1, 1))
	if y is not None:
		y = numpy.asarray(y).astype(numpy.float64)
		if len(y.shape) < 2:
			## must be 2D with order of [row, column], EVEN FOR 1D DATA!
			y = y.reshape((-1, 1))
	return X, y
