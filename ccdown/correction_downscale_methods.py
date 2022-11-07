# An additional set of methods for downscaling
# all methods should implement at least fit, predict, and score, to fit in the same framework as the sklearn methods

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.stats as st
import seaborn as sns
import sklearn.metrics
import sklearn.ensemble
import sliced


class quantile_mapping(object):
	# map quantiles from input data (GCM) to observation

	def __init__(self):
		self.correction = None
		self.trained = False

	def fit(self, modeled, obs):
		obs = np.array(obs)
		obs = obs[np.logical_not(np.isnan(obs))]
		data_size = obs.shape[0]
		# Set bins edges
		obs_sorted = np.sort(obs).squeeze()
		train_sorted = np.sort(np.array(modeled).squeeze())

		train_interp = interp1d(np.arange(1, train_sorted.shape[0] + 1), train_sorted)
		train_sorted = train_interp(np.linspace(1, train_sorted.shape[0], data_size))

		self.correction = np.array(train_sorted - obs_sorted)
		self.trained = True
		return self.correction

	def predict(self, model_timeseries):
		model_timeseries = np.array(model_timeseries).reshape(-1, ).tolist()
		model_sorted = sorted(model_timeseries)
		adjusted = []
		for i in range(len(model_sorted)):
			rank = model_sorted.index(model_timeseries[i]) / float(len(model_sorted))
			rank = int(rank * self.correction.shape[0])
			adjusted.append(model_timeseries[i] - self.correction[rank])
		return np.array(adjusted)

	def score(self, x_test, y_true):
		y_pred = self.predict(x_test)
		return sklearn.metrics.r2_score(y_true, y_pred)


class two_step_random_forest(object):
	"""This is a two-part random forest model, for use especially with precipitation downscaling. This first part is a
	classifier, to distinguish between dry and wet days. The second kicks in on the wet days, to produce a value
	for how much precipitation falls on those days.
	trace_value: value to count everything below as "no precipitation """

	def __init__(self, trace_value=0.01):
		self.pop_rf = sklearn.ensemble.RandomForestClassifier()
		self.qpf_rf = sklearn.ensemble.RandomForestRegressor()
		self.trace_value = trace_value

	def fit(self, x_train, y_train):
		# recode to binary yes no on rain
		y_train_class = np.array([1 if a > self.trace_value else 0 for a in y_train]).astype(int)

		# train classifier
		self.pop_rf.fit(x_train, y_train_class)

		# train regressor on the non-dry days
		y_train_qpf = y_train[y_train_class != 0]
		x_train_qpf = x_train[y_train_class != 0]
		print(x_train_qpf.shape, y_train_qpf.shape)
		self.qpf_rf.fit(x_train, y_train)

	def predict(self, x_test):
		dry_days = self.pop_rf.predict(x_test)
		qpf = self.qpf_rf.predict(x_test)

		qpf[dry_days == 0] = 0
		return qpf

	def score(self, x_test, y_true):
		y_test = self.predict(x_test)
		return sklearn.metrics.r2_score(y_true, y_test)


class sir_downscaling(object):
	# downscale using the SIR sufficient dimension reduction method
	# uses https://github.com/joshloyal/sliced.git for SIR
	def __init__(self, n_directions=20):
		self.sir = sliced.SlicedInverseRegression(n_directions=n_directions)

	def fit(self, x_train, y_train):
		self.sir.fit(x_train, y_train)

	def predict(self, x_test):
		return self.sir.transform(x_test)

	def score(self, x_test, y_true):
		y_test = self.predict(x_test)
		return sklearn.metrics.r2_score(y_true, y_test)


def test_prcp_model():
	# Simple tests for the methods above using random data
	sim = np.random.rand(10000, 50)
	obs = np.random.normal(1, 2, size=10000) + np.random.normal(2, 3, size=10000)
	test = np.random.rand(10000, 50)

	# sns.kdeplot(sim, label = 'sim')
	sns.kdeplot(obs, label='obs')
	# sns.kdeplot(test, label = 'test')

	model = two_step_random_forest()
	model.fit(sim, obs)
	sim_transformed = model.predict(sim)
	test_transformed = model.predict(test)
	print(model.score(sim, obs))

	print('sim trasnformed, test transformed, obs mean')
	print(np.nanmean(sim_transformed), np.nanmean(test_transformed), np.nanmean(obs))

	sns.kdeplot(sim_transformed, label='sim_transformed', linestyle='dashed')
	sns.kdeplot(test_transformed, label='test_transformed', linestyle='dotted')
	plt.legend()
	plt.show()


def test_temperature_model():
	sim = np.random.normal(3, 4, size=10000)
	obs = np.random.normal(4, 2, size=10000)  # +np.random.normal(2,3, size = 10000)
	test = np.random.normal(10, 4, size=10000)

	sns.kdeplot(sim, label='sim')
	sns.kdeplot(obs, label='obs')
	sns.kdeplot(test, label='test')

	sim = sim.reshape(-1, 1)
	test = test.reshape(-1, 1)

	qmap = quantile_mapping()
	# qmap = sir_downscaling(n_directions=2)
	qmap.fit(sim, obs)
	sim_transformed = qmap.predict(sim)
	test_transformed = qmap.predict(test)
	print(qmap.score(sim_transformed, obs))

	print('sim transformed, test transformed, obs mean')
	print(np.nanmean(sim_transformed), np.nanmean(test_transformed), np.nanmean(obs))

	sim_transformed = sim_transformed.reshape(-1, )
	test_transformed = test_transformed.reshape(-1, )
	sns.kdeplot(sim_transformed, label='sim_transformed', linestyle='dashed')
	sns.kdeplot(test_transformed, label='test_transformed', linestyle='dotted')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	# test_temperature_model()
	test_prcp_model()
