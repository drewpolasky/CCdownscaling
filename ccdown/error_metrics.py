import numpy as np


def calc_bias(model_data, obs_data):
	# Calculate average error
	bias = np.nanmean(obs_data - model_data)
	return bias


if __name__ == '__main__':
	pass
