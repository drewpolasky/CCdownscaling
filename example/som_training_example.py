# This is a simple example to show the process of choosing the size of the SOM map
import sys

import xarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src import som_downscale, utilities


def som_size_selection(downscaling_target='precip', station_id='725300-94846'):
	# dictionary of variables and pressure levels to use from the reanalysis data
	input_vars = {'air': 850, 'rhum': 850, 'uwnd': 700, 'vwnd': 700, 'hgt': 500}
	station_data = pd.read_csv('./data/stations/' + station_id + '.csv')
	station_data = station_data.replace(to_replace=[99.99, 9999.9], value=np.nan)
	reanalysis_data = xarray.open_mfdataset('./data/models/*NCEP*')

	stations_info = pd.read_csv('./data/stations/stations.csv')
	station_info = stations_info.loc[stations_info['stationID'] == station_id]
	station_lat = station_info['LAT'].values
	station_lon = station_info['LON'].values

	# load the reanalysis precipitation
	if downscaling_target == 'precip':
		rean_precip = xarray.open_mfdataset('./data/reanalysis/prate_1976-2005_NCEP_midwest.nc')
		# convert from Kg/m^2/s to mm/day
		rean_precip['prate'] = rean_precip['prate'] * 86400
		rean_precip = rean_precip['prate'].sel(lat=station_lat, lon=station_lon, method='nearest').values
		rean_precip = np.squeeze(rean_precip)
	elif downscaling_target == 'max_temp':
		rean_precip = xarray.open_mfdataset('./data/models/air_1976-2005_NCEP_midwest.nc')
		rean_precip = rean_precip['air'].sel(level=1000, lat=station_lat, lon=station_lon, method='nearest').values
		# convert K to C
		rean_precip = rean_precip - 273.15
		rean_precip = np.squeeze(rean_precip)

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
	window = 2
	lat_index = np.argmin(np.abs(reanalysis_data['lat'].values - station_lat))
	lon_index = np.argmin(np.abs(reanalysis_data['lon'].values - station_lon))
	reanalysis_data = reanalysis_data.isel({'lat': slice(lat_index - window, lat_index + window + 1),
											'lon': slice(lon_index - window, lon_index + window + 1)})

	input_data = []
	for var in input_vars:
		var_data = reanalysis_data.sel(level=input_vars[var])[var].values
		var_data = var_data.reshape(var_data.shape[0], var_data.shape[1] * var_data.shape[2])
		input_data.append(var_data)
	input_data = np.concatenate(input_data, axis=1)
	input_data = np.array(input_data)

	# Drop days with NaN values for the observation:
	hist, rean_precip = utilities.remove_missing(hist_data, rean_precip)
	hist_data, input_data = utilities.remove_missing(hist_data, input_data)

	input_data, input_means, input_stdevs = utilities.normalize_climate_data(input_data)

	# split train and test sets:
	# train_split = int(round(input_data.shape[0]*0.8))
	train_split = 8765  # split out the first 24 years for the training data, last 6 years for the test set
	training_data = input_data[0:train_split, :]
	train_hist = hist_data[0:train_split]
	test_data = input_data[train_split:, :]
	test_hist = hist_data[train_split:]
	rean_precip_train = rean_precip[0:train_split]
	rean_precip_test = rean_precip[train_split:]
	print(training_data.shape, test_data.shape)

	sizes = [[2, 3], [3, 4], [4, 5], [5, 7], [6, 8], [7, 9], [9, 11], [11, 13], [13, 15]]
	quant_errors = []
	topo_errors = []
	for size in sizes:
		som = som_downscale.som_downscale(som_x=size[0], som_y=size[1], batch=512, alpha=0.1, epochs=50)
		som.fit(training_data, train_hist, seed=1)
		quant_errors.append(som.quantization_error(training_data))
		topo_errors.append(som.topograpical_error(training_data))

	size_names = [str(size[0]) + 'x' + str(size[1]) for size in sizes]
	plt.plot(size_names, quant_errors)
	plt.xlabel('SOM size')
	plt.ylabel('Quantization Error')
	plt.savefig('./example_figures/quant_errors_sizes.png')
	plt.show()

	plt.plot(size_names, topo_errors)
	plt.xlabel('SOM size')
	plt.ylabel('Topograpical Error')
	plt.savefig('./example_figures/topo_errors_sizes.png')
	plt.show()


if __name__ == '__main__':
	som_size_selection()
