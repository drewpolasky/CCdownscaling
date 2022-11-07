# Automated methods for variable selection
# All methods will return ranked list of variables in order of importance,
# some (SIR, PCA) will also return a dimension reduced dataset
import sklearn.ensemble
import xarray
import pandas as pd
import numpy as np
from sklearn import decomposition
import sliced

from ccdown import utilities


def select_vars(input_data, target_data, method, labels=None):
	"""
	Wrapper function for variable selection methods
	@param input_data: array, uses the same format as inputs to the downscaling methods
	@param target_data: target_data: 1-D array
	@param method: string, implemented methods are: sir, pca, randomforest
	@param labels: names of the input variables. Should match the shape of input_data[0]
	@return:
	"""

	if method.lower() == 'sir':
		transformed, most_important, weights = SIR_selection(input_data, target_data)
	elif method.lower() == 'pca':
		transformed, most_important = PCA_selection(input_data)
	elif method.lower() in ['randomforest', 'rf']:
		importances, most_important = randomforest_selection(input_data, target_data)
		transformed = None
	else:
		print('variable selection method not recognized')
		exit()
	if labels is not None:
		important_labels = [labels[i] for i in most_important]
		important_labels.reverse()
	else:
		important_labels = None
	print(important_labels)
	return transformed, important_labels


def SIR_selection(input_data, target_data, n_components=None):
	"""
	Rank variables and dimension reduce using Sliced Inverse Regression
	@param input_data: array, uses the same format as inputs to the downscaling methods
	@param n_components: number of components to calculate. Defaults to the number of variables in the input data
	@param target_data: target_data: 1-D array
	@return:
	"""
	if n_components is None:
		n_components = input_data.shape[-1]
	sir = sliced.SlicedInverseRegression(n_directions=n_components)
	sir.fit(input_data, target_data)
	weights = np.abs(np.sum((sir.directions_.T * sir.eigenvalues_).T, axis=0))
	most_important = [i for i in range(n_components)]
	sorted_weights, most_important = zip(*sorted(zip(weights, most_important)))
	output_data = sir.transform(input_data)
	return output_data, most_important, sorted_weights


def PCA_selection(input_data, n_components=None):
	"""
	Rank variables and dimension reduce using Principal Component Analysis
	@param input_data: array, uses the same format as inputs to the downscaling methods
	@param n_components: number of components to calculate. Defaults to the number of variables in the input data
	@return:
	"""
	if n_components is None:
		n_components = input_data.shape[-1]
	pca = decomposition.PCA(n_components=n_components)
	pca.fit(input_data)
	most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_components)]
	output_data = pca.transform(input_data)
	return output_data, most_important


def randomforest_selection(input_data, target_data):
	"""
	Rank variables using the sklearn impurity-based variable importance for random forest.
	@param input_data: array, uses the same format as inputs to the downscaling methods
	@param target_data: 1-D array
	@return:
	"""
	rf = sklearn.ensemble.RandomForestRegressor()
	rf.fit(input_data, target_data)
	importances = rf.feature_importances_
	most_important = [i for i in range(input_data.shape[-1])]
	sorted_importances, most_important = zip(*sorted(zip(importances, most_important)))
	return sorted_importances, most_important


def organize_labeled_data(input_vars, reanalysis_data, window=0):
	# simple function to generate a list of the names of the input variables
	input_data = []
	labels = []
	for var in input_vars:
		if type(input_vars[var]) is not list:
			input_vars[var] = [input_vars[var]]
		for level in input_vars[var]:
			var_data = reanalysis_data.sel(level=level)[var].values
			if window != 0:
				var_labels = np.array([[var + '_' + str(level) + '_' + str(i) + '_' + str(j) for i in range(-window, window+1)] for j in range(-window, window+1)])
			else:
				var_labels = np.array([[var + '_' + str(level)]])
			var_labels = var_labels.reshape(1, var_data.shape[1] * var_data.shape[2])
			var_data = var_data.reshape(var_data.shape[0], var_data.shape[1] * var_data.shape[2])
			input_data.append(var_data)
			labels.append(var_labels)
	input_data = np.concatenate(input_data, axis=1)
	input_data = np.array(input_data)
	labels = np.array(labels)
	labels = np.concatenate(labels, axis=1).squeeze()
	return input_data, labels


def test_selection():
	#Test case for the variable selection code using the o'hare example
	station_id = '725300-94846'
	downscaling_target = 'precip'
	input_vars = {'air': [850, 700, 500, 300], 'rhum': [850, 700, 500, 300],
				  'uwnd': [850, 700, 500, 300], 'vwnd': [850, 700, 500, 300],
				  'hgt': [850, 700, 500, 300]}
	station_data = pd.read_csv('../example/data/stations/' + station_id + '.csv')
	station_data = station_data.replace(to_replace=[99.99, 9999.9], value=np.nan)
	reanalysis_data = xarray.open_mfdataset('../example/data/models/*NCEP*')

	stations_info = pd.read_csv('../example/data/stations/stations.csv')
	station_info = stations_info.loc[stations_info['stationID'] == station_id]
	station_lat = station_info['LAT'].values
	station_lon = station_info['LON'].values

	# select the station data to match the time from of the reanalysis data
	start = reanalysis_data['time'][0].values
	end = reanalysis_data['time'][-1].values
	station_data['time'] = pd.to_datetime(station_data['date'], format='%Y-%m-%d')
	date_mask = ((station_data['time'] >= start) & (station_data['time'] <= end))
	station_data = station_data[date_mask]

	hist_data = station_data[downscaling_target].values
	# Convert units, F to C for temperature, in/day to mm/day for precip
	if downscaling_target == 'max_temp':
		hist_data = (hist_data - 32) * 5 / 9
	if downscaling_target == 'precip':
		hist_data = hist_data * 25.4
	# For just a single grid point:
	# reanalysis_data = reanalysis_data.sel(lat = station_lat, lon = station_lon, method='nearest')
	# To use multiple grid points in a window around the location:
	window = 0
	lat_index = np.argmin(np.abs(reanalysis_data['lat'].values - station_lat))
	lon_index = np.argmin(np.abs(reanalysis_data['lon'].values - station_lon))
	reanalysis_data = reanalysis_data.isel({'lat': slice(lat_index - window, lat_index + window + 1),
											'lon': slice(lon_index - window, lon_index + window + 1)})

	input_data, labels = organize_labeled_data(input_vars, reanalysis_data)
	# Drop days with NaN values for the observation:
	hist_data, input_data = utilities.remove_missing(hist_data, input_data)
	input_data, input_means, input_stdevs = utilities.normalize_climate_data(input_data)

	# Run each of the variable selection methods for comparison
	select_vars(input_data, hist_data, method='SIR', labels=labels)
	select_vars(input_data, hist_data, method='PCA', labels=labels)
	select_vars(input_data, hist_data, method='RF', labels=labels)


if __name__ == '__main__':
	test_selection()
