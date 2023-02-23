import numpy as np
import statsmodels.api as sm

def calc_bias(model_data, obs_data):
	# Calculate average error
	bias = np.nanmean(obs_data - model_data)
	return bias

def calc_aic(model_data, obs_data):
	# Calculate the Akaike information criterion
	reg = sm.OLS(model_data, obs_data).fit()
	AIC = reg.aic
	return AIC

if __name__ == '__main__':
	pass
