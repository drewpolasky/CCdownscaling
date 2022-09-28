# Methods to set up alternate train/test splits, for testing the downscaling
# methods on different climates than they were trained on.

import numpy as np
import pandas as pd
import xarray

from src import utilities


def simple_split(x_data, y_data, rean_y=None, train_split=8760):
	train_x = x_data.isel(time=slice(0, train_split))
	train_y = y_data.isel(time=slice(0, train_split))
	test_x = x_data.isel(time=slice(train_split, -1))
	test_y = y_data.isel(time=slice(train_split, -1))
	if rean_y is not None:
		rean_train = rean_y.isel(time=slice(0, train_split))
		rean_test = rean_y.isel(time=slice(train_split, -1))
		return train_x, train_y, test_x, test_y, rean_train, rean_test
	else:
		return train_x, train_y, test_x, test_y


def select_max_target_years(x_data, y_data, y_target, time_period='year', split=0.8, rean_data=None):
	"""

	@param rean_data:
	@param split:
	@param x_data:
	@param y_data:
	@param y_target: str, max or min
	@param time_period: must match pandas/xarray time periods: day, month, year, etc
	"""
	max_periods = y_data.groupby('time.' + time_period).mean('time')
	split_value = max_periods.quantile(q=split)
	if y_target == 'max':
		test_periods = max_periods[time_period][max_periods >= split_value].values
		train_periods = max_periods[time_period][max_periods < split_value].values
	elif y_target == 'min':
		test_periods = max_periods[time_period][max_periods <= split_value].values
		train_periods = max_periods[time_period][max_periods > split_value].values
	else:
		print('y target not recognized, enter either "max" or "min"')
		exit()
	x_train = x_data.sel(time=(x_data.time.dt.year.isin(train_periods)))
	x_test = x_data.sel(time=(x_data.time.dt.year.isin(test_periods)))
	y_train = y_data.sel(time=(y_data.time.dt.year.isin(train_periods)))
	y_test = y_data.sel(time=(y_data.time.dt.year.isin(test_periods)))

	train_average = y_train.mean('time')
	test_average = y_test.mean('time')
	print('train period average: ', train_average.values)
	print('test period averaged: ', test_average.values)
	if rean_data is not None:
		rean_train = rean_data.sel(time=(rean_data.time.dt.year.isin(train_periods)))
		rean_test = rean_data.sel(time=(rean_data.time.dt.year.isin(test_periods)))
		return x_train, x_test, y_train, y_test, rean_train, rean_test
	else:
		return x_train, x_test, y_train, y_test


def select_season_train_test(x_data, y_data, train_dates, test_dates, rean_data=None):
	"""

	@param rean_data:
	@param x_data: xarray dataframe
	@param y_data: array-like, matching shape of x_data
	@param train_dates: list of dates, str or datetime
	@param test_dates: list of dates, str or datetime
	"""
	x_train = x_data.sel(time=x_data.time.isin(train_dates))
	x_test = x_data.sel(time=x_data.time.isin(test_dates))
	y_train = y_data.sel(time=y_data.time.isin(train_dates))
	y_test = y_data.sel(time=y_data.time.isin(test_dates))
	train_average = y_train.mean('time', skipna=True)
	test_average = y_test.mean('time', skipna=True)
	print('train period average: ', train_average.values)
	print('test period averaged: ', test_average.values)
	if rean_data is not None:
		rean_train = rean_data.sel(time=rean_data.time.isin(train_dates))
		rean_test = rean_data.sel(time=rean_data.time.isin(test_dates))
		return x_train, x_test, y_train, y_test, rean_train, rean_test
	else:
		return x_train, x_test, y_train, y_test


def test_train_test_splits():
	station_id = '725300-94846'
	downscaling_target = 'max_temp'
	# downscaling_target = 'precip'
	input_vars = {'air': [1000, 500], 'rhum': [850], 'uwnd': [700], 'vwnd': [700], 'hgt': [500]}
	station_data = pd.read_csv('../example/data/stations/' + station_id + '.csv')
	station_data = station_data.replace(to_replace=[99.99, 9999.9], value=np.nan)
	reanalysis_data = xarray.open_mfdataset('../example/data/models/*NCEP*', combine='by_coords')

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

	# train/test splits:
	# add dates to the observation data, using the same dates as the reanalysis data
	dates = reanalysis_data['time']
	hist_data = xarray.DataArray(data=hist_data, dims=['time'], coords={'time': dates})
	#train_data, test_data, train_hist, test_hist = select_max_target_years(reanalysis_data, hist_data, 'max',
	#																	   time_period='year', split=0.8)
	#print(train_data, test_data)

	# season select
	train_dates = utilities.generate_dates_list('3/1', '5/31', list(range(1976, 2006)))
	test_dates = utilities.generate_dates_list('6/1', '8/31', list(range(1976, 2006)))
	train_data, test_data, train_hist, test_hist = select_season_train_test(reanalysis_data, hist_data, train_dates,
																			test_dates)
	print(train_data)
	print(test_data)


if __name__ == '__main__':
	test_train_test_splits()
