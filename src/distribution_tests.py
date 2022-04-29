# Distribution based tests for evaluating downscaling methods.
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


def ks_testing(modelData, histData):
	"""
	Calculate the 2 sample kolmogorov-smirnov test for a distribution of historical and modeled data
	:param: modelData: array-like
	:param: hisData: array-like
	"""
	ks_results = scipy.stats.ks_2samp(modelData, histData)
	return ks_results


def pdf_skill_score(dist1, dist2, n=20, plot=False, non_zero=False):
	"""
	calculate the PDF skill score (Perkins et al. 2007,
	https://journals.ametsoc.org/view/journals/clim/20/17/jcli4253.1.xml) between 2 distributions dist1 and dist2
	are lists or numpy arrays, n is an int for the number of bins in the pdf non_zero is bool option to only use
	non-zero values, which can be useful for assessing precipitation
	"""
	score = 0
	dist1 = np.sort(np.array(dist1))
	dist2 = np.sort(np.array(dist2))

	minVal = min([dist1.min(), dist2.min()])
	maxVal = max([dist1.max(), dist2.max()])
	bins = np.linspace(minVal, maxVal, n)

	hist1, bin_edges = np.histogram(dist1, bins=bins, density=False)
	hist2, bin_edges = np.histogram(dist2, bins=bins, density=False)
	hist1 = hist1 / len(dist1)
	hist2 = hist2 / len(dist2)

	if plot:
		plt.hist([dist1, dist2], bins=bins, density=True, label=['model', 'obs'])
		plt.legend()
		plt.show()

	for i in range(len(bins) - 1):
		score += min(hist1[i], hist2[i])
	return score


def threshold_frequency(model_data, obs_data, threshold):
	"""
	Calculate the relative frequency of exceeding a threshold between the modeled and observed data. For example,
	setting a threshold value of 0.01 for precipitation data will return the difference in percentage of days with
	trace or less precipitation
	"""

	model_above = len(np.where(np.array(model_data) > threshold)[0]) / len(model_data)
	obs_above = len(np.where(np.array(model_data) > threshold)[0]) / len(obs_data)
	return obs_above - model_above


if __name__ == '__main__':
	pass
