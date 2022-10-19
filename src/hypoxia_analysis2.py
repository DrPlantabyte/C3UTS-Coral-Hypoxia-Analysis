import os, numpy, pandas, scipy, dmba, statsmodels
from os import path
from typing import Tuple, List
from copy import deepcopy

from dmba import AIC_score
from pandas import DataFrame, Series
from numpy import ndarray, nan, float64
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from data_loading import scan_dir
from respiration_models import RespirationModelABC, BestPoly, ModelS1, ModelS2, ModelS3, MichaelisMenten, FixedMinMaxScaler
from transformers import SelectColumns

# note: convert to CSV with
# $ in2csv --write-sheets "-" -f xlsx ./Colony\ 2\ -\ A.\ cf.\ microphthalma\ Hypoxia\ Data\ \(Inner\ vs\ Outer\).xlsx

def main():
	data_dir = '../data/Nicole-Data'
	# print(os.listdir(data_dir))
	output_dir = '../run/output/Nicole-Data'
	SHOW_FIGS = False
	if not path.isdir(output_dir):
		os.makedirs(output_dir)
	for data_file in scan_dir(data_dir, regex=r'.*/Colony.*\.csv'):
		print('Processing "%s"...' % data_file)
		fname = path.basename(data_file)
		colony = fname.split('-')[0].replace('Colony', '').strip()
		organism = fname.split('-')[1].split('Hypoxia Data')[0].strip()
		rep_num = fname.split('_')[1].replace('.csv', '').strip()
		title = '%s (col. %s)' % (organism, colony)
		subtitle = 'Sample %s' % rep_num
		# _, title, subtitle = data_file.replace('.csv', '').split('_')
		hypoxia_data = parse_hypoxia_csv(data_file, header_row=2).dropna()
		real_data = numpy.asarray(hypoxia_data[['PO2 (%)', 'VO2 (mg O2/hr)']], dtype=float64)
		# normalized_data = data_normalizer.fit_transform(hypoxia_data[['PO2 (%)', 'VO2 (mg O2/hr)']])
		## using cubic interpolation to estimate max vO2, and manually setting max pO2 to 100%
		ymax=numpy.polyval(numpy.polyfit(real_data[:, 0], real_data[:, 1], 3), 100.)
		data_normalizer = FixedMinMaxScaler(min_values=[0,0], max_values=[100, ymax])
		normalized_data = data_normalizer.fit_transform(real_data)
		best_model, model_info = model_hypoxia_curve(xdata=normalized_data[:, 0], ydata=normalized_data[:, 1])
		analyze_model(
			model=best_model, model_info=model_info, hypoxia_data=hypoxia_data,
			data_normalizer=data_normalizer, specimen=title, experiment=subtitle,
			figfile=path.join(output_dir, 'AIC-selection_Nicole-Data_%s_%s.pdf' % (title, subtitle)), show=SHOW_FIGS
		)
	print('...Done!')


def model_hypoxia_curve(xdata, ydata) -> Tuple[RespirationModelABC, DataFrame]:
	##
	models: List[RespirationModelABC] = [
		BestPoly(max_order=12),
		ModelS1(),
		ModelS2(),
		ModelS3(),
		MichaelisMenten()
	]
	##
	model_results = []
	best_aic = numpy.inf
	best_model = None
	for model in models:
		model.fit(xdata, ydata)
		aic_score = model.AIC_score(xdata, ydata)
		model_results.append(Series({
			'Model Name': model.name(),
			'Model': model,
			'AIC': aic_score,
		}))
		if aic_score < best_aic:
			best_aic = aic_score
			best_model = model
	df = DataFrame(model_results)
	evidence_ratios = calculate_AIC_evidence_ratios(df['AIC'])
	er_df = DataFrame(evidence_ratios, columns=['ER %s' % c for c in df['Model Name'].values])
	df = pandas.concat([df, er_df], axis=1)
	return best_model, df

def analyze_model(
		model: RespirationModelABC, model_info: DataFrame, hypoxia_data: DataFrame, data_normalizer: FixedMinMaxScaler, specimen: str, experiment: str,
		figfile=None, show=False
):
	xcol_name = 'PO2 (%)'
	ycol_name='VO2 (mg O2/hr)'
	this_model_info: Series = model_info[model_info['Model Name'] == model.name()].iloc[0]
	xdata_real = numpy.asarray(hypoxia_data[xcol_name]); ydata_real = numpy.asarray(hypoxia_data[ycol_name])
	normalized_data = data_normalizer.transform(hypoxia_data[[xcol_name, ycol_name]])
	xdata = normalized_data[:, 0]; ydata = normalized_data[:, 1]
	ypred = numpy.clip(model.predict(xdata).ravel(), 0., ydata.max()) # clip to datya range to prevent bestpoly model from running up to infinity
	dydx = model.dydx(xdata)
	rel_stdev = (ypred - ydata).std() / ypred
	xmax_real = data_normalizer.max_values[0]
	xmax_norm = data_normalizer.transform([[xmax_real,0]])[0,0]
	ymax = numpy.clip(model.predict(xmax_norm)[0,0], 0., ydata.max()) # cap maximum to keep bestpoly models from eating the axese
	ymean = numpy.nanmean(ypred) / xmax_norm
	# unnormalized_prediction = data_normalizer.inverse_transform(numpy.asarray(list(zip(xdata, ypred))))
	# ypred_real = unnormalized_prediction[:,1]; xdata_real = unnormalized_prediction[:,0]
	rho = ypred / xdata - dydx.ravel()
	max_reg_idx = numpy.argmax(rho)
	min_reg_idx = numpy.argmin(rho)
	ymean_idx = arg_first_above(ypred, ymean, reverse=True)
	relmax_reg_idxs = argrelextrema(rho, numpy.greater)[0]
	Tdir = numpy.nanmean(rho)
	Tabs = numpy.nanmean(numpy.abs(rho))
	Tpos = (Tabs + Tdir) / 2
	Tneg = (Tabs - Tdir) / 2
	half_max_idx = numpy.argmin(numpy.abs(ypred - (ymax/2)))
	PTpos_idx = arg_first_above(rho, Tpos, reverse=True)
	## confidence intervals
	vO2_sigma = numpy.std(ydata - ypred)
	rho_sigma = rho.mean() * vO2_sigma / ypred.mean()
	# vO2_lower_ci, vO2_upper_ci = bootstrap_prediction_confidence_interval(xdata, ydata, model, percentile=95)
	# vO2_lower_ci, vO2_upper_ci = stats.norm.interval(confidence=0.95, loc=ypred, scale=stats.sem(ydata))
	# rho_lower_ci, rho_upper_ci = stats.norm.interval(confidence=0.95, loc=rho, scale=stats.sem(ydata))
	vO2_lower_ci = ypred - 2*vO2_sigma; vO2_upper_ci = ypred + 2*vO2_sigma
	rho_lower_ci = rho - 2*rho_sigma; rho_upper_ci = rho + 2*rho_sigma
	Pmdr_idx = arg_first_above(rho_lower_ci, 0, reverse=True)
	##
	print(this_model_info.to_markdown())
	chart_table = DataFrame([Series({
		'specimen': specimen,
		'replicate': experiment,
		'num pts': len(xdata),
		'min PO2 ref': data_normalizer.min_values[0],
		'max PO2 ref': data_normalizer.max_values[0],
		'min VO2 ref': data_normalizer.min_values[1],
		'max VO2 ref': ymax,
		'VO2 mean': ypred.mean(),
		'best model': model.name(),
		'ER_S1': this_model_info['ER S1'],
		'ER_S2': this_model_info['ER S2'],
		'ER_S3': this_model_info['ER S3'],
		'ER_sk': this_model_info['ER SK'],
		'ER_bestpoly': this_model_info['ER Best Poly'],
		'Pc-max': xdata_real[max_reg_idx],
		'Pc-max_1': xdata_real[relmax_reg_idxs[0]] if len(relmax_reg_idxs) > 0 else nan,
		'Pc-max_2': xdata_real[relmax_reg_idxs[1]] if len(relmax_reg_idxs) > 1 else nan,
		'Pc-max_3': xdata_real[relmax_reg_idxs[2]] if len(relmax_reg_idxs) > 2 else nan,
		'Pc-max_4': xdata_real[relmax_reg_idxs[3]] if len(relmax_reg_idxs) > 3 else nan,
		'Pc-min': xdata_real[min_reg_idx],
		## Note: total regulation is the integral of rho across the range (x0, x1) where rho is positive or negative,
		## divided by the length of that range (x1-x0)
		'Tpos': Tpos,
		'Tneg': Tneg,
		'P50': xdata_real[half_max_idx], # PO2 at 50% Vmax O2
		# R is the area under the Vo2 vs Po2 curve and expressed as a percentage of the total possible area
		'R': 100. * (numpy.nansum(ypred) / (ymax * len(ypred))),
		'Ymean': ymean,
		'Pymean': xdata_real[ymean_idx],
		'PTpos': xdata_real[PTpos_idx],
		'Pmdr': xdata_real[Pmdr_idx],
	})])

	# chart_table = model_info[model_info['Model Name'] == model.name()].transpose(copy=True)
	chart_table = chart_table.transpose(copy=True)
	print(chart_table.to_markdown())
	##
	if show or figfile is not None:
		from matplotlib import pyplot
		from matplotlib.axes import Axes
		from matplotlib.figure import Figure
		# fig, axes = pyplot.subplots(nrows=2, ncols=1, sharex=True)
		# top_ax: Axes = axes[0]; btm_ax: Axes = axes[1]
		fig: Figure = pyplot.figure(figsize=(8, 6))
		## NOTE: any errors in grid alignment will generate a confusing "AttributeError: 'NoneType' object has no attribute 'dpi_scale_trans'" error
		tbl_ax: Axes = pyplot.subplot2grid(shape=(2, 3), loc=(0, 0), rowspan=2, colspan=1)
		top_ax: Axes = pyplot.subplot2grid(shape=(2, 3), loc=(0, 1), rowspan=1, colspan=2)
		btm_ax: Axes = pyplot.subplot2grid(shape=(2, 3), loc=(1, 1), rowspan=1, colspan=2, sharex=top_ax)
		#
		top_ax.set_title('Hypoxia Plot +/- 95% confidence interval')
		btm_ax.set_title('Profile Plot +/- 95% confidence interval')
		btm_ax.set_xlabel('pO2 (%)')
		top_ax.set_ylabel('vO2 (rel. units)')
		btm_ax.set_ylabel('rho')
		top_ax.grid()
		btm_ax.grid()
		#
		## plot data and model
		full_plot_x = numpy.linspace(0, xmax_real, 256)
		full_plot_x_norm = numpy.linspace(0, xmax_norm, 256)
		top_ax.plot(xdata_real, ydata, 'k.')
		top_ax.plot(full_plot_x, numpy.clip(model.predict(full_plot_x_norm).ravel(), 0, ydata.max()), 'k-')
		## plot zero-reg reference line
		top_ax.plot([0, xmax_real], [0, ymax], 'g--')
		## plot mean Vo2/Po2 point
		half_max_x = xdata_real[half_max_idx]
		half_max_y = ypred[half_max_idx]
		top_ax.plot([0, half_max_x, half_max_x], [half_max_y, half_max_y, 0], 'r-')
		top_ax.plot([half_max_x], [half_max_y], 'ro', markerfacecolor='none')
		## using bootstrap prediction confidence intervals
		top_ax.plot(xdata_real, vO2_upper_ci, 'k--')
		top_ax.plot(xdata_real, vO2_lower_ci, 'k--')
		top_ax.plot([xdata_real[max_reg_idx], xdata_real[max_reg_idx]], [ypred[max_reg_idx], 0], 'y-x')
		#
		btm_ax.plot(xdata_real, rho, 'k-')
		btm_ax.plot([0, xmax_real], [0, 0], 'g--')
		btm_ax.plot(xdata_real, rho_upper_ci, 'k--')
		btm_ax.plot(xdata_real, rho_lower_ci, 'k--')
		btm_ax.plot([xdata_real[max_reg_idx], xdata_real[max_reg_idx]], [rho[max_reg_idx], 0], 'y-x')
		btm_ax.plot([xdata_real[Pmdr_idx], xdata_real[Pmdr_idx]], [rho_lower_ci[Pmdr_idx], 0], 'm-x')
		#
		# f: Figure = pyplot.gcf()
		# text_pos = numpy.asarray([0.1, 1.])
		# for k in info:
		# 	text_pos[1] -= 0.04
		# 	fig.text(*text_pos, "%s: %s" % (k, info[k]), font='mono')

		tbl_ax.set_axis_off()
		tbl_plt = tbl_ax.table(
			loc='center', bbox=[0,0,1,1],
			cellText=numpy.asarray(chart_table),
			rowLabels=chart_table.index,
			colLabels=None,
			cellLoc='left', rowLoc='right',
			edges='open'
		)
		tbl_plt.set_fontsize(14)
		tbl_plt.scale(1.5, 1.5)
		#
		fig.tight_layout()
		if figfile is not None:
			pyplot.savefig(figfile)
		if show:
			pyplot.show()
		fig.clf()
		pyplot.close(fig)

def bootstrap_prediction_confidence_interval(X: ndarray, y: ndarray, model: RespirationModelABC, percentile=95, num_bootstraps=1000) -> Tuple[ndarray, ndarray]:
	upper_pct = 0.5*(percentile+100)
	lower_pct = 100-upper_pct
	bootstrap_model = deepcopy(model)
	n = len(X)
	bootstrap_values = numpy.zeros((num_bootstraps, n))
	for i in range(0, num_bootstraps):
		## From "Practical Statistics for Data Scientists" O'Reilly book
		# 1. take a bootstrap sample
		bootstrap_X, bootstrap_Y = bootstrap_resample(X.ravel(), y.ravel())
		# 2. fit the regression and predict the new value
		bootstrap_model.fit(bootstrap_X.reshape((-1, 1)), bootstrap_Y.reshape((-1, 1)))
		bootstrap_ypred = bootstrap_model.predict(X).ravel()
		# 3. take a single residual at random (residual is predicted - real) and add it to the predicted value (new random sample for each point)
		residuals = bootstrap_ypred - y
		bootstrap_ypred = bootstrap_ypred + residuals.take(numpy.random.randint(0, n, size=n))
		# 4. repeat 1000 times
		bootstrap_values[i] = bootstrap_ypred
		# 5. take 2.5 and 97.5 percentile values for 95% interval
	return numpy.percentile(bootstrap_values, lower_pct, axis=0), numpy.percentile(bootstrap_values, upper_pct, axis=0)


def bootstrap_resample(xdata: ndarray, ydata: ndarray) -> Tuple[ndarray, ndarray]:
	n = len(xdata)
	indices = numpy.random.randint(0, n, size=2*n)
	return xdata.take(indices), ydata.take(indices)

def calculate_AIC_evidence_ratios(AICs: ndarray) -> ndarray:
	# see Wagenmakers and Farrel, AIC model selection using Akaike weights, Psychonomic Bulletin & Review, 2004, 11 (1), 192-196
	AICs = numpy.asarray(AICs)
	dAICs = AICs - AICs.min()
	wAICs = numpy.exp(-0.5 * dAICs) / numpy.sum(numpy.exp(-0.5 * dAICs))
	return numpy.outer(wAICs, 1./wAICs)

def arg_first_above(x: ndarray, threshold: float, reverse=False) -> int:
	x = numpy.asarray(x)
	if reverse:
		return len(x) - arg_first_above(numpy.flip(x), threshold, reverse=False) - 1
	for i in range(0, len(x)):
		if x[i] > threshold:
			return i
	return len(x)-1


def parse_hypoxia_csv(filepath, header_row=1,) -> DataFrame:
	row_skip = header_row
	if row_skip < 0: row_skip = None
	df= pandas.read_csv(filepath, skiprows=row_skip)
	print(list(df.columns))
	if 'O2 mg/L' in df.columns:
		df.insert(3, 'O2 (mg/L)', df['O2 mg/L'])
	return df[['Time (s)', 'Time (mins)', 'PO2 (%)', 'O2 (mg/L)', 'O2  -  control (mg/L)', 'VO2 (mg O2/hr)']]

#
if __name__ == '__main__':
	main()
