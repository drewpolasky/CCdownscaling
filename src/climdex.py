
import numpy as np
import pandas as pd
import xarray

from simple_pyclimdex.src import prcp, max_temp, min_temp
from src import utilities

def print_indices(outputs, func_list):
	"""

	@param outputs:
	@param func_list:
	@return:
	"""
	np.set_printoptions(precision=1)
	names = []
	for func in func_list:
		name = func[0].__name__
		for key in func[1]:
			name += '_' + key + '_' + str(func[1][key])
		names.append(name)

	for i in range(len(func_list)):
		print(names[i], round(np.nanmean(outputs[i]),2))


def calc_climdex(data, target, reference_data = None):
	"""

	@param data:
	@param target:
	@param reference_data:
	@return:
	"""
	if target in ['prcp', 'precip']:
		func_list = [[prcp.prcp_mean,{}], [prcp.rx1_day,{}],[prcp.rx5_day,{}], [prcp.r95p,{}],[prcp.r95p,{'threshold':99}],[prcp.sdii,{}],[prcp.cdd,{}],[prcp.cwd,{}],[prcp.r10mm,{}], [prcp.r10mm,{'threshold':20}]]
	elif target in ['tmax', 'max_temp']:
		func_list = [[max_temp.tx_mean,{}],[ max_temp.tx_min,{}], [max_temp.tx_max,{}], [max_temp.su25,{}],[ max_temp.id0,{}], [max_temp.tx90p,{'reference_tmax':reference_data}], [max_temp.tx10p,{'reference_tmax':reference_data}], [max_temp.wsdi,{'reference_tmax':reference_data}]]
	elif target in ['tmin','min_temp']:
		func_list = [[min_temp.tn_mean,{}], [min_temp.tn_min,{}], [min_temp.tn_max,{}], [min_temp.tr20,{}], [min_temp.tn90p,{'reference_tmin':reference_data}], [min_temp.tn10p,{'reference_tmin':reference_data}], [min_temp.csdi,{'reference_tmin':reference_data}]]
	else:
		print('unknown target variable')
		exit()
	outputs = []
	for i in range(len(func_list)):
		#print(func.__name__)
		output = func_list[i][0](data, **func_list[i][1])
		outputs.append(output)
	return outputs, func_list

def test_climdex():
	station_id = '725300-94846'
	downscaling_target = 'max_temp'
	#downscaling_target = 'precip'
	#downscaling_target = 'min_temp'
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

	hist_data = utilities.remove_leap_days(hist_data, start_year=1976)

	outputs, func_list = calc_climdex(hist_data, downscaling_target)
	print_indices(outputs, func_list)



if __name__ == '__main__':
	test_climdex()






