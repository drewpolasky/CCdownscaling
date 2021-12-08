#This is an example usage of the downscaling package, 
#using GSOD station data for Chicago Midway airport


import sys
import random 

import xarray
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/gpfs/group/jle7/default/adp29/downscaling_package')
import som_downscale
import correction_downscale_methods
import utilities
import distribution_tests
import error_metrics
import correlation_metrics

#for reproducability
seed = 1
random.seed(seed)

def downscale_example(downscaling_target = 'precip', station_id = '725300-94846'):

	#dictionary of variables and pressure levels to use from the reanalysis data
	input_vars = {'air': 850, 'rhum': 850, 'uwnd': 700, 'vwnd': 700, 'hgt': 500}
	station_data = pd.read_csv('./data/stations/' + station_id + '.csv')
	station_data = station_data.replace(to_replace=[99.99, 9999.9], value=np.nan)
	reanalysis_data = xarray.open_mfdataset('./data/models/*NCEP*')

	stations_info = pd.read_csv('./data/stations/stations.csv')
	station_info = stations_info.loc[stations_info['stationID'] == station_id]
	station_lat = station_info['LAT'].values
	station_lon = station_info['LON'].values

	#load the reanalysis precipitation
	if downscaling_target == 'precip':
		rean_precip = xarray.open_mfdataset('./data/reanalysis/prate_1976-2005_NCEP_midwest.nc')
		#convert from Kg/m^2/s to mm/day
		rean_precip['prate'] = rean_precip['prate'] * 86400
		rean_precip = rean_precip['prate'].sel(lat = station_lat, lon = station_lon, method='nearest').values
		rean_precip = np.squeeze(rean_precip)
	elif downscaling_target == 'max_temp':
		rean_precip =  xarray.open_mfdataset('./data/models/air_1976-2005_NCEP_midwest.nc')
		rean_precip = rean_precip['air'].sel(level = 1000, lat = station_lat, lon = station_lon, method='nearest').values
		#convert K to C
		rean_precip = rean_precip - 273.15
		rean_precip = np.squeeze(rean_precip)

	#select the station data to match the time from of the reanalysis data
	start = reanalysis_data['time'][0].values
	end = reanalysis_data['time'][-1].values
	station_data['time'] = pd.to_datetime(station_data['date'], format='%Y-%m-%d')
	date_mask = ((station_data['time'] >= start) & (station_data['time']  <= end))
	station_data = station_data[date_mask]

	hist_data = station_data[downscaling_target].values
	#Convert units, F to C for temperature, in/day to mm/day for precip
	if downscaling_target == 'max_temp':
		hist_data = (hist_data - 32)*5/9 
	if downscaling_target == 'precip':
		hist_data = hist_data * 25.4
	#For just a single grid point:
	#reanalysis_data = reanalysis_data.sel(lat = station_lat, lon = station_lon, method='nearest')
	#To use multiple grid points in a window around the location:
	window = 2
	lat_index = np.argmin(np.abs(reanalysis_data['lat'].values - station_lat))
	lon_index = np.argmin(np.abs(reanalysis_data['lon'].values - station_lon))
	reanalysis_data = reanalysis_data.isel({'lat':slice(lat_index-window,lat_index+window+1),'lon':slice(lon_index-window,lon_index+window+1)})

	training_data = []
	for var in input_vars:
		var_data = reanalysis_data.sel(level = input_vars[var])[var].values
		var_data = var_data.reshape(var_data.shape[0], var_data.shape[1]*var_data.shape[2])
		training_data.append(var_data)
	training_data = np.concatenate(training_data, axis = 1)
	training_data = np.array(training_data)

	#Drop days with NaN values for the observation:
	hist, rean_precip = utilities.remove_missing(hist_data, rean_precip)
	hist_data, training_data = utilities.remove_missing(hist_data, training_data)

	training_data, training_means, training_stdevs = utilities.normalize_climate_data(training_data)

	#intialize the different methods
	som = som_downscale.som_downscale(som_x = 7, som_y = 5, batch = 512, alpha = 0.1, epochs = 50)
	rf_two_part = correction_downscale_methods.two_step_random_forest()
	random_forest = sklearn.ensemble.RandomForestRegressor()
	qmap = correction_downscale_methods.quantile_mapping()

	#train
	som.fit(training_data, hist_data, seed = 1)
	random_forest.fit(training_data, hist_data)
	rf_two_part.fit(training_data, hist_data)
	qmap.fit(rean_precip, hist_data)

	#generate outputs from the training data 
	som_output = som.predict(training_data)
	random_forest_output = random_forest.predict(training_data)
	rf_two_part_output = rf_two_part.predict(training_data)

	qmap_output = qmap.predict(rean_precip)

	#Include the reanalysis precipitation as an undownscaled comparison
	names = ['SOM','Random Forest','RF Two Part','Qmap','NCEP']
	outputs = [som_output, random_forest_output, rf_two_part_output, qmap_output, rean_precip]

	#names = ['SOM','Random Forest','NCEP']
	#outputs = [som_output, random_forest_output, rean_precip]
	#run analyses for the downscaled outputs

	#first, the som specific plots
	freq, avg, dry = som.node_stats()
	#ax = som.heat_map(training_data, annot=avg)
	#plt.show()
	i = 0 
	index_range = (window*2 + 1)**2
	for var in input_vars:
		start_index = i*index_range
		end_index = (i+1)*index_range
		fig, ax = som.plot_nodes(weights_index=(start_index, end_index), means = training_means[start_index:end_index], stdevs = training_stdevs[start_index:end_index], cmap='bwr')
		fig.suptitle(var)
		fig.savefig('example_figures/SOM_nodes_NCEP_' + var +'.png')
		plt.close()
		i+=1

	#next, the various skill metric scores
	scores = {}
	np.set_printoptions(precision=2, suppress=True)
	i = 0
	for output in outputs:
		pdf_score = distribution_tests.pdf_skill_score(output, hist_data)
		ks_stat, ks_probs = distribution_tests.ks_testing(output, hist_data)
		rmse = sklearn.metrics.mean_squared_error(hist_data, output, squared=False)
		bias = error_metrics.bias(output, hist_data)

		print(round(pdf_score,3), round(ks_stat,2), round(rmse,2), round(bias,2))
		scores[names[i]] = [round(pdf_score,3), round(ks_stat,2), round(rmse,2), round(bias,2)]
		i += 1

	#finally, some plots comparing the outputs
	i = 0
	fig, ax = plt.subplots(nrows=1, ncols=1)
	for output in outputs:
		sns.kdeplot(output, label = names[i] + ' ' + str(scores[names[i]][0]), ax = ax)
		i += 1
	sns.kdeplot(hist_data, color = 'k', lw=2.0, label = 'Obs', ax = ax)
	plt.show()

	plot_histogram(outputs, names, hist_data)


def plot_histogram(outputs, names, hist_data):
	fig, ax = plt.subplots(nrows=1, ncols=1)
	bin_starts = np.array([0, 0.01, 0.05, .1, .25, .75, 2, 10]) * 25.4
	outputs.append(hist_data)
	names.append('Obs')
	ax.hist(outputs, bins=bin_starts, label = names, density = True, rwidth = .4, log = True)
	plt.xscale("log")
	logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
	ax.xaxis.set_major_formatter(logfmt)
	plt.xlabel(r'PRCP ($10^x$ mm/day)')
	ax.yaxis.set_major_formatter(logfmt)
	plt.ylabel(r'Frequency ($10^x$)')
	ax.tick_params(axis='both')
	ax.legend()
	plt.show()



if __name__ == '__main__':
	downscale_example()