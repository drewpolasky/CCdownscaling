import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import acf


def plot_kde(outputs, names, hist_data, scores=None, downscaling_target=None, ax=None, fig=None):
	i = 0
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1)
	for output in outputs:
		if scores is not None:
			sns.kdeplot(output, label=names[i] + ' ' + str(scores[names[i]][0]), ax=ax)
		else:
			sns.kdeplot(output, label=names[i], ax=ax)
		i += 1
	sns.kdeplot(hist_data, color='k', lw=2.0, label='Obs', ax=ax)
	if downscaling_target == 'max_temp':
		plt.xlabel(r'Daily Max Temperature ($^\circ$C)')
	elif downscaling_target == 'precip':
		plt.xlabel(r'PRCP ($10^x$ mm/day)')
	return fig, ax


def plot_histogram(outputs, names, hist_data, fig=None, ax=None, downscaling_target=None, y_label=True):
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1)
	bin_starts = np.array([0, 0.01, .1, .25, .75, 2, 10]) * 25.4
	outputs.append(hist_data)
	names.append('Wet Years Obs')
	ax.hist(outputs, bins=bin_starts, label=names, density=True, rwidth=.4, log=True)
	plt.xscale("log")
	logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
	#ax.xaxis.set_major_formatter(logfmt)
	ax.set_xlabel(r'PRCP (mm/day)')
	#ax.yaxis.set_major_formatter(logfmt)
	if y_label:
		ax.set_ylabel(r'Frequency')
	ax.tick_params(axis='both')
	ax.legend()
	return fig, ax


def plot_autocorrelation(outputs, names, hist_data, nlags=10):
	# plot the autocorrelation function for the different downscaling outputs.
	fig, ax = plt.subplots(nrows=1, ncols=1)
	i = 0
	for output in outputs:
		auto_corr = acf(output, nlags=nlags)
		ax.plot(auto_corr, label=names[i])
		i += 1

	obs_corr = acf(hist_data, nlags=nlags)
	ax.plot(obs_corr, label='Obs', color='k')
	ax.set_xlabel('Lag Time (days)')
	ax.set_ylabel('Correlation')
	ax.legend(ncol=2)
	return fig, ax


if __name__ == '__main__':
	pass
