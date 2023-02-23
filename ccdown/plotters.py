import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import acf


def plot_kde(outputs, names, hist_data, scores=None, downscaling_target=None, ax=None, fig=None, cumulative=False):
	i = 0
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1)
	for output in outputs:
		if scores is not None:
			sns.kdeplot(output, label=names[i] + ' ' + str(scores[names[i]][0]), ax=ax, cumulative=cumulative)
		else:
			sns.kdeplot(output, label=names[i], ax=ax, cumulative=cumulative)
		i += 1
	sns.kdeplot(hist_data, color='k', lw=2.0, label='Obs', ax=ax, cumulative=cumulative)
	if downscaling_target == 'max_temp':
		plt.xlabel(r'Daily Max Temperature ($^\circ$C)')
	elif downscaling_target == 'precip':
		plt.xlabel(r'Precipitation ($10^x$ mm/day)')
	return fig, ax


def plot_histogram(outputs, names, hist_data, fig=None, ax=None, downscaling_target=None, y_label=True):
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1)
	bin_starts = np.array([0, 0.01, .1, .25, .75, 2, 10]) * 25.4
	outputs.append(hist_data)
	#names.append('Wet Years Obs')
	ax.hist(outputs, bins=bin_starts, label=names, density=True, rwidth=.4, log=True)
	plt.xscale("log")
	logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
	ax.xaxis.set_major_formatter(logfmt)
	ax.set_xlabel(r'Precipitation (mm/day)')
	ax.yaxis.set_major_formatter(logfmt)
	if y_label:
		ax.set_ylabel(r'Frequency')
	ax.tick_params(axis='both')
	ax.legend()
	return fig, ax


def plot_bargraph(outputs, names, hist_data, fig=None, ax=None, downscaling_target=None, y_label=True, figsize = None, **kwargs):
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
	bin_starts = np.array([0, 0.01, .1, .25, .75, 2, 10]) * 25.4
	outputs.append(hist_data)
	names.append('Obs')
	i = 0
	width = 1 / (len(names) + 1)
	tick_spots = np.array([j for j in range(len(bin_starts) - 1)])
	for output in outputs:
		output_hist = np.histogram(output, bin_starts, density=True)
		ax.bar(tick_spots + i * width, output_hist[0], label=names[i], log=True, width=width, **kwargs)
		i += 1

	ax.set_xlabel(r'Precipitation (mm/day)')
	xticks = plt.xticks()[0]
	bin_width = (max(xticks) - min(xticks)) / (len(bin_starts) - 1)
	ax.set_xticks(tick_spots + width * len(names)/2, np.round(bin_starts[:-1], 2))
	if y_label:
		ax.set_ylabel(r'Frequency')
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


def scatter_plot(outputs, names, hist_data):
	# plot a scatter plot of the downscaled values against the observed values for the same day
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))
	i = 0
	for output in outputs:
		mask = ~np.isnan(hist_data) & ~np.isnan(output)
		slope, intercept, r_value, p_value, std_err = stats.linregress(hist_data[mask], output[mask])
		ax.plot(hist_data[mask], output[mask], '.', label=names[i] + r' R$^2$: ' + str(round(r_value ** 2, 2)))
		i += 1

	ax.set_xlabel(r'Observed Data ($^\circ$C)')
	ax.set_ylabel(r'Downscaled Data ($^\circ$C)')
	ax.legend(loc='upper left')
	return fig, ax



if __name__ == '__main__':
	pass
