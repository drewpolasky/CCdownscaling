# Helper functions

import numpy as np


def normalize_climate_data(model_data, means=None, stdevs=None):
	"""
	Takes the means and stdev for a historical/reanalysis data and calculates z-scores of the passed in climate data
	based on those. if means and standard deviations are given, calculates the z-scores from those values, rather than
	the model_data, so that the different periods of data can be compared If means and stdevs are None, scale to the
	mean and stdev of the data :param model_data: example shape - (10950x175) for 30 years with 7 variables on 5x5
	grid :param means: shape should match all but first axis of model_data :param stdevs: shape same as means :return:
	scaled model_data; returns same stdevs and means if they are passed in, and the stdev and means of the climate
	data if none are passed in.
	"""
	if means is None:
		means = np.mean(model_data, axis=0)
		stdevs = np.std(model_data, axis=0)
		model_data = model_data - means
		model_data = model_data / stdevs
		return model_data, means, stdevs
	else:
		model_means = np.mean(model_data, axis=0)
		model_stdevs = np.std(model_data, axis=0)
		model_data = model_data - means
		model_data = model_data / stdevs
		return model_data, model_means, model_stdevs


def select_season(data, start_date, end_date):
	"""
	selects portions of years in the input data, for looking at individual seasons. Assumes the data starts on january 1
	Note: assumes leap days are not present
	:param data:		list of day values
	:param start_date:  day of year to start including, can be negative to wrap around the start of the year 
	:param end_date:    day of year to end selection
	"""
	if start_date >= 0:
		station_values = [data[i] for i in range(len(data)) if
						  start_date <= i % 365 <= end_date]
	else:
		station_values = [data[i] for i in range(len(data)) if
						  start_date <= i % 365 <= end_date or i % 365 >= start_date % 365]
	return station_values


def remove_leap_days(data, start_year):
	"""remove leap days from an array of data
	:param data: array
	:param start_year: int, year the data starts from
	"""
	newClimateData = []
	numLeapDays = int(0)
	flag = True
	start_year = int(start_year)
	for i in range(len(data)):
		if (i - numLeapDays) % 365 == 31 + 28 and flag:
			if (start_year + (i - numLeapDays) // 365) % 4 == 0 and (start_year + (i - numLeapDays) // 365) % 100 != 0:
				numLeapDays += 1
				flag = False
			elif (start_year + (i - numLeapDays) // 365) % 4 == 0 and (
					start_year + (i - numLeapDays) // 365) % 1000 == 0:
				numLeapDays += 1
				flag = False
			else:
				newClimateData.append(data[i])
		else:
			newClimateData.append(data[i])
			flag = True
	return newClimateData


def remove_missing(historicalData, climateData):
	"""remove days in the historical or climate data that are missing information"""
	newHist = []
	newClim = []
	numRemoved = 0
	for i in range(len(historicalData)):
		if historicalData[i] == 'NaN' or np.isnan(historicalData[i]) or np.isnan(np.sum(climateData[i])):
			numRemoved += 1
		else:
			newHist.append(historicalData[i])
			newClim.append(climateData[i])
	return np.array(newHist), np.array(newClim)


if __name__ == '__main__':
	pass
