
import time
import random
import math
import pickle

import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import sklearn.ensemble

from src.tensorflow_som.tf_som import SelfOrganizingMap as SOM


class som_downscale(object):
	def __init__(self, som_x, som_y, batch, alpha, epochs, num_procs=1):
		self.epochs = epochs
		self.som_x = som_x
		self.som_y = som_y
		self.batch = batch
		self.alpha = alpha
		self.num_procs = num_procs
		self.clusters = {}
		self.node_model_type = 'random'
		self.node_models = []

	def fit(self, model_timeseries, obs_timeseries, seed=1):
		"""
		Because the SOM itself is an unsupervised method, a single som can be trained for multiple downscale
		locations that all fall in the same input data grid point. The differences are in the PDFs fit to each SOM
		node. Fit, in order to match the API of the other methods, both trains the SOM, and fits the PDF,
		but these steps can be broken out into multiple steps with the map and fit_pdf functions
		"""
		self.map(model_timeseries, seed=seed)
		self.fit_pdfs(model_timeseries, obs_timeseries)
		if self.node_model_type.lower() == 'random_forest':
			self.train_rf_node_models(model_timeseries, obs_timeseries)

	def map(self, model_timeseries, seed=1):
		graph = tf.Graph()
		with graph.as_default():
			tf.set_random_seed(seed)
			session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=False,
				intra_op_parallelism_threads=self.num_procs,
				inter_op_parallelism_threads=0))

			dims = len(model_timeseries[0])

			# put the training data into the tf input format. see https://www.tensorflow.org/programmers_guide/datasets
			trainData = tf.data.Dataset.from_tensor_slices(model_timeseries.astype(np.float32))
			trainData = trainData.repeat()
			trainData = trainData.batch(self.batch)
			iterator = trainData.make_one_shot_iterator()
			nextElement = iterator.get_next()

			som = SOM(m=self.som_y, n=self.som_x, dim=dims, batch_size=self.batch, initial_learning_rate=self.alpha,
					  graph=graph, session=session, input_tensor=nextElement, max_epochs=self.epochs)

			init_op = tf.compat.v1.global_variables_initializer()
			session.run([init_op])
			trainTime = time.time()
			som.train(num_inputs=len(model_timeseries), save_map=True)
			self.output_weights = som.output_weights
			print('time to train for ' + str(self.epochs) + ' epochs: ', time.time() - trainTime)
		self.trained = True

	def save(self, save_path):
		pickle.dump(self.output_weights, open(save_path + 'weights.p', 'wb'))
		pickle.dump(self.clusters, open(save_path + 'clusters.p', 'wb'))

	def load(self, save_path):
		self.output_weights = pickle.load(open(save_path + 'weights.p', 'rb'))
		self.clusters = pickle.load(open(save_path + 'clusters.p', 'rb'))

	def fit_pdfs(self, model_timeseries, obs_timeseries):
		for i in range(len(model_timeseries)):
			dayVector = model_timeseries[i]
			min_index = min([j for j in range(len(self.output_weights))],
							key=lambda x: np.linalg.norm(dayVector - self.output_weights[x]))

			if min_index not in self.clusters:
				self.clusters[min_index] = []

			self.clusters[min_index].append(obs_timeseries[i])

	def train_rf_node_models(self, model_timeseries, obs_timeseries):
		node_days = [[] for i in range(len(self.clusters))]
		for i in range(len(model_timeseries)):
			dayVector = model_timeseries[i]
			min_index = min([j for j in range(len(self.output_weights))],
							key=lambda x: np.linalg.norm(dayVector - self.output_weights[x]))
			node_days[min_index].append(i)

		for i in range(len(self.clusters)):
			x_train = model_timeseries[node_days[i]]
			y_train = obs_timeseries[node_days[i]]
			self.node_models.append(sklearn.ensemble.RandomForestRegressor())
			self.node_models[i].fit(x_train, y_train)

	def predict(self, model_timeseries):
		output = []
		for i in range(len(model_timeseries)):
			dayVector = model_timeseries[i]
			min_index = min([i for i in range(len(self.output_weights))],
							key=lambda x: np.linalg.norm(dayVector - self.output_weights[x]))
			if self.node_model_type == 'random':
				dayValue = self.generate_obs(min_index)
			else:
				dayValue = self.generate_rf_obs(min_index, dayVector)
			output.append(dayValue)
		return np.array(output)

	def generate_obs(self, index, rank=None):
		"""
		generates a daily observation based on the given values. If rank is given, the observation will be taken at
		that quantile, otherwise it will be randomly generated. rank: rank of CDF to generate at. default to random if
		None :return: float value of observation
		"""
		if rank is None:
			rank = random.random()
		# following Crane and Hewitson (2006), create a spline fit CDF to the cluster values, and select the value at
		# the specified (or random) rank value CDF code adapted from:
		# https://stackoverflow.com/questions/24575869/read-file-and-plot-cdf-in-python
		clusterValues = [i for i in self.clusters[index] if not np.isnan(i)]
		data_size = len(clusterValues)
		# Set bins edges
		data_set = sorted(set(clusterValues))
		bins = np.append(data_set, data_set[-1] + 1)

		# Use the histogram function to bin the data
		counts, bin_edges = np.histogram(clusterValues, bins=bins, density=False)
		counts = counts.astype(float) / data_size

		# Create the cdf
		cdf = np.cumsum(counts)
		x = bin_edges[0:-1]
		y = cdf

		if max(clusterValues) != min(clusterValues):
			f = interp1d(x, y, kind='slinear')
			xnew = np.arange(min(clusterValues), max(clusterValues) - 0.01, 0.1)
			min_index = np.unravel_index(np.argmin(np.abs(rank - f(xnew))), f(xnew).shape)
			return xnew[min_index]
		else:
			return (max(clusterValues))  # if all the cluster values are the same

	def generate_rf_obs(self, index, day_vector):
		"""
		Rather than using a random selection from the set of days in each som node, train an additional model,
		one the days that fall on that som node, and use that to generate the day observations
		"""
		day_model = self.node_models[index]
		day_value = day_model.predict(day_vector)
		return day_value

	def quantization_error(self, model_timeseries):
		# distance between data points and their nearest nodes
		distances = []
		for i in range(len(model_timeseries)):
			dayVector = model_timeseries[i]
			min_index = min([i for i in range(len(self.output_weights))],
							key=lambda x: np.linalg.norm(dayVector - self.output_weights[x]))
			distances.append(np.linalg.norm(dayVector - self.output_weights[min_index]))

		quant_errors = np.mean(distances)
		return quant_errors

	def topograpical_error(self, model_timeseries):
		"""
		look at the best and second-best nodes, and see how far apart they are -- closer is general better,
		indicating the topology of the input space is being maintained in the 2-d graph
		"""
		te = 0
		for i in range(len(model_timeseries)):
			dayVector = model_timeseries[i]
			node_list = sorted([j for j in range(len(self.output_weights))],
							   key=lambda x: np.linalg.norm(dayVector - self.output_weights[x]))
			min_index, second_index = node_list[0], node_list[1]
			if self.is_adjacent(min_index, second_index):
				te += 0
			else:
				te += 1
		topo_errors = (te / (i + 1))
		return topo_errors

	def is_adjacent(self, index_one, index_two):
		# identify if two indices for a linearized som graph are adjacent to one another
		adjacents = [index_one + 1, index_one - 1, index_one + self.som_x, index_one - self.som_x]
		if index_two in adjacents:
			return True
		else:
			return False

	def plot_nodes(self, weights_index=None, size_x=None, size_y=None, stdevs=None, means=None,
				   fmt=matplotlib.ticker.FormatStrFormatter('%.0f'), cmap=cm.gist_ncar, **kwargs):
		"""
		plots the grid of SOM nodes, with contours for a selected range of weights, intended to correspond to a single
		variable over a spatial window.
		:param weights_index:  Optionally pass in subset of SOM weights to plot
		:param stdevs:	Optionally pass in standard deviations to de-normalize weights, which are trained on z-score data
		:param means: As above, but the means for the training data
		:param cmap: Specify a color map to use when plotting
		further kwargs will be passed to pyplot contourf
		:return:
		"""
		n = self.som_x
		m = self.som_y

		fig, axes = plt.subplots(m, n, figsize=(11, 9), sharex=True, sharey=True)

		# optionally select out the value for the variable of focus.
		if weights_index is not None:
			plot_weights = self.output_weights[:, weights_index[0]:weights_index[1]]
		else:
			plot_weights = self.output_weights
		zs = {}
		for j in range(len(plot_weights)):
			loc = (j // n, j % n)
			zs[loc] = []
			for i in range(len(plot_weights[0])):
				if stdevs is not None and means is not None:
					zs[loc].append(plot_weights[j][i] * stdevs[i] + means[i])
				else:
					zs[loc].append(plot_weights[j][i])

		minValue = np.min(list(zs.values()))
		maxValue = np.max(list(zs.values()))

		if size_x is None:
			size_x = round(math.sqrt(len(plot_weights[0])))
		if size_y is None:
			size_y = round(math.sqrt(len(plot_weights[0])))

		for j in range(len(plot_weights)):
			loc = (j // n, j % n)
			plot_zs = np.array(zs[loc]).reshape(size_x, size_y)
			axis = axes[loc[0], loc[1]]
			contours = axis.contourf(plot_zs, levels=np.linspace(minValue, maxValue, 20), cmap=cmap, vmin=minValue,
									 vmax=maxValue, **kwargs)

		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.6])
		cbar = fig.colorbar(contours, cax=cbar_ax, format=fmt)
		return fig, axes, cbar

	def node_stats(self):
		"""
		Calculates information about the nodes of the SOM. SOM must be trained before running returns: average value
		of node, frequency of node in training data, percent of days with a value of 0 (mainly for precipitation,
		the number of dry days)
		"""
		n = self.som_x
		m = self.som_y
		gridAvg = np.zeros((m, n))
		gridFreq = np.zeros((m, n))
		gridStDev = np.zeros((m, n))
		gridDryDays = np.zeros((m, n))
		for cluster in self.clusters:
			loc = [cluster // n, cluster % n]
			gridAvg[loc[0], loc[1]] = float(np.nanmean(np.array(self.clusters[cluster])))
			gridFreq[loc[0], loc[1]] = len(self.clusters[cluster])
			gridStDev[loc[0], loc[1]] = np.nanmax(self.clusters[cluster])
			gridDryDays[loc[0], loc[1]] = self.clusters[cluster].count(0) / float(len(self.clusters[cluster]))
		return gridFreq, gridAvg, gridDryDays

	def heat_map(self, model_timeseries, **kwargs):
		"""
		Plots the relative frequency of each SOM node in the provided data. SOM must be trained first. 
		"""
		indices = np.zeros((self.som_y, self.som_x))
		for i in range(len(model_timeseries)):
			dayVector = model_timeseries[i]
			min_index = min([i for i in range(len(self.output_weights))],
							key=lambda x: np.linalg.norm(dayVector - self.output_weights[x]))
			indices[min_index // self.som_x, min_index % self.som_x] += 1
		ax = sns.heatmap(indices, **kwargs)
		return ax


def test_som():
	np.random.seed(1)
	random.seed(1)
	tf.set_random_seed(1)

	train_data = np.random.normal(3, 4, size=(10000, 25))
	train_obs = np.random.normal(3, 4, size=(10000))

	test_data = np.random.normal(3, 4, size=(10000, 25))
	test_obs = np.random.normal(3, 4, size=(10000))

	som = som_downscale(3, 5, 128, 0.01, 25)
	som.map(train_data)
	som.fit_pdfs(train_data, train_obs)
	print(som.output_weights.shape)
	predicted = som.predict(test_data)
	print(predicted.shape)
	print(som.node_stats())


# ax = som.heat_map(train_data, annot=True)
# plt.show()
# fig, ax = som.plot_nodes()
# plt.show()

if __name__ == '__main__':
	test_som()
