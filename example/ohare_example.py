# This is an example usage of the downscaling package,
# using GSOD station data for Chicago Midway airport

import random

import xarray
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf

from src import correction_downscale_methods, distribution_tests, error_metrics, som_downscale, utilities, \
    train_test_splits, climdex

# for reproducibility
seed = 1
random.seed(seed)


def downscale_example(downscaling_target='precip', station_id='725300-94846'):
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

    # Identify the most important variables, and do dimension reduction, if desired
    ##TODO

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

    # Drop leap days for ease of use:
    hist_data = utilities.remove_leap_days(hist_data, start_year=1976)
    rean_precip = utilities.remove_leap_days(rean_precip, start_year=1976)
    reanalysis_data = utilities.remove_xarray_leap_days(reanalysis_data)

    # split train and test sets:
    dates = reanalysis_data['time']
    hist_data = xarray.DataArray(data=hist_data, dims=['time'], coords={'time': dates})
    rean_precip = xarray.DataArray(data=rean_precip, dims=['time'], coords={'time': dates})
    # with a simple split:
    train_data, train_hist, test_data, test_hist, rean_precip_train, rean_precip_test = train_test_splits.simple_split(
        reanalysis_data, hist_data, rean_precip)

    # selecting the highest precip/temperature years:
    train_data, test_data, train_hist, test_hist, rean_precip_train, rean_precip_test = train_test_splits.select_max_target_years(
       reanalysis_data, hist_data, 'max', time_period='year', split=0.8, rean_data=rean_precip)

    # training on the spring, testing on the summer:
    train_dates = utilities.generate_dates_list('3/1', '5/31', list(range(1976, 2006)))
    test_dates = utilities.generate_dates_list('6/1', '8/31', list(range(1976, 2006)))
    #train_data, test_data, train_hist, test_hist, rean_precip_train, rean_precip_test = train_test_splits.select_season_train_test(
    #	reanalysis_data, hist_data, train_dates, test_dates, rean_data=rean_precip)

    input_train_data = []
    input_test_data = []
    for var in input_vars:
        var_data = train_data.sel(level=input_vars[var])[var].values
        var_data = var_data.reshape(var_data.shape[0], var_data.shape[1] * var_data.shape[2])
        input_train_data.append(var_data)
        var_test_data = test_data.sel(level=input_vars[var])[var].values
        var_test_data = var_test_data.reshape(var_test_data.shape[0], var_test_data.shape[1] * var_test_data.shape[2])
        input_test_data.append(var_test_data)
    input_train_data = np.concatenate(input_train_data, axis=1)
    input_train_data = np.array(input_train_data)
    input_test_data = np.concatenate(input_test_data, axis=1)
    input_test_data = np.array(input_test_data)

    # normalize (z-score) the reanalysis data, for use in downscaling methods
    train_data, input_means, input_stdevs = utilities.normalize_climate_data(input_train_data)
    test_data, input_test_means, input_test_stdevs = utilities.normalize_climate_data(input_test_data,
                                                                                      means=input_means,
                                                                                      stdevs=input_stdevs)
    #print(train_data.shape, test_data.shape, train_hist.shape, test_hist.shape)

    # Drop days with NaN values for the observation:
    hist, rean_precip_train = utilities.remove_missing(train_hist, rean_precip_train)
    train_hist, train_data = utilities.remove_missing(train_hist, train_data)
    #test, rean_precip_test = utilities.remove_missing(test_hist, rean_precip_test)
    #test_hist, test_data = utilities.remove_missing(test_hist, test_data)

    print(train_data.shape, test_data.shape, train_hist.shape, test_hist.shape)
    print(np.nanpercentile(train_hist, 90))
    print(np.nanpercentile(test_hist, 90))

    # initialize the different methods
    som = som_downscale.som_downscale(som_x=7, som_y=5, batch=512, alpha=0.1, epochs=50)
    rf_two_part = correction_downscale_methods.two_step_random_forest()
    random_forest = sklearn.ensemble.RandomForestRegressor()
    qmap = correction_downscale_methods.quantile_mapping()
    linear = sklearn.linear_model.LinearRegression()

    # train
    som.fit(train_data, train_hist, seed=1)
    random_forest.fit(train_data, train_hist)
    rf_two_part.fit(train_data, train_hist)
    linear.fit(train_data, train_hist)
    qmap.fit(rean_precip_train, train_hist)

    # generate outputs from the test data
    som_output = som.predict(test_data)
    random_forest_output = random_forest.predict(test_data)
    rf_two_part_output = rf_two_part.predict(test_data)
    linear_output = linear.predict(test_data)
    qmap_output = qmap.predict(rean_precip_test)

    # generate outputs from the train data for comparision
    som_train_output = som.predict(train_data)
    random_forest_train_output = random_forest.predict(train_data)
    rf_two_part_train_output = rf_two_part.predict(train_data)
    linear_train_output = linear.predict(train_data)
    qmap_train_output = qmap.predict(rean_precip_train)

    # Include the reanalysis precipitation as an undownscaled comparison
    names = ['SOM', 'Random Forest', 'RF Two Part', 'Linear', 'Qmap', 'NCEP']
    outputs = [som_output, random_forest_output, rf_two_part_output, linear_output, qmap_output, rean_precip_test]

    # names = ['SOM','Random Forest','NCEP']
    # outputs = [som_output, random_forest_output, rean_precip]
    # run analyses for the downscaled outputs

    # first, the som specific plots
    freq, avg, dry = som.node_stats()
    ax = som.heat_map(train_data, annot=avg)
    plt.yticks(rotation=0)
    plt.savefig('example_figures/ohare_heatmap_highSplit_5x7_' + downscaling_target + '.png')
    # plt.show()
    plt.close()
    i = 0
    index_range = (window * 2 + 1) ** 2
    for var in input_vars:
        start_index = i * index_range
        end_index = (i + 1) * index_range
        fig, ax, cbar = som.plot_nodes(weights_index=(start_index, end_index), means=input_means[start_index:end_index],
                                       stdevs=input_stdevs[start_index:end_index], cmap='bwr')
        for axis in ax.flatten():
            axis.set_xticks([])
            axis.set_yticks([])

        for axis, col in zip(ax[-1], range(0, som.som_x)):
            axis.set_xlabel(col, size='large')
        for axis, row in zip(ax[:, 0], range(0, som.som_y)):
            axis.set_ylabel(row, rotation=0, size='large')
        # fig.suptitle(var)
        units = {'air': '(K)', 'rhum': '(%)', 'uwnd': r'(ms$^{-1}$)', 'vwnd': r'(ms$^{-1}$)', 'hgt': '(m)'}
        cbar.set_label(var.capitalize() + ' ' + units[var], rotation='horizontal', labelpad=20)
        fig.savefig('example_figures/SOM_nodes_highSplit_NCEP_' + var + '.png')
        # plt.show()
        plt.close()
        i += 1

    # next, the various skill metric scores
    scores = {}
    np.set_printoptions(precision=2, suppress=True)
    i = 0
    for output in outputs:
        noNan_test_hist, noNan_output = utilities.remove_missing(test_hist, output)
        pdf_score = distribution_tests.pdf_skill_score(noNan_output, noNan_test_hist)
        ks_stat, ks_probs = distribution_tests.ks_testing(noNan_output, noNan_test_hist)
        rmse = sklearn.metrics.mean_squared_error(noNan_test_hist, noNan_output, squared=False)
        bias = error_metrics.calc_bias(noNan_output, noNan_test_hist)

        print(names[i], round(pdf_score, 3), round(ks_stat, 2), round(rmse, 2), round(bias, 2))
        scores[names[i]] = [round(pdf_score, 3), round(ks_stat, 2), round(rmse, 2), round(bias, 2)]
        i += 1

    # finally, some plots comparing the outputs
    fig, ax = plot_kde(outputs, names, test_hist, scores, downscaling_target)
    plt.savefig('example_figures/ohare_maxYears_highSplit_' + downscaling_target + '_methods_compare_kde.png')
    #plt.show()

    fig, ax = plot_autocorrelation(outputs, names, test_hist)
    plt.savefig('example_figures/ohare_maxYears_highSplit_' + downscaling_target + '_methods_compare_autocorr.png')
    #plt.show()

    fig, ax = plot_histogram(outputs, names, test_hist)
    plt.savefig('example_figures/ohare_maxYears_highSplit_' + downscaling_target + '_methods_compare_histogram.png')
    #plt.show()
    plt.close('all')

    for i in range(len(outputs)):
        climdex_values, func_list = climdex.calc_climdex(outputs[i], downscaling_target, reference_data = train_hist)
        print()
        print(names[i])
        climdex.print_indices(climdex_values, func_list)

    fig, axes = plt.subplots(1,3, figsize=(15,8), sharex=True, sharey=True)
    outputs = [[som_train_output, som_output, train_hist],[random_forest_train_output, random_forest_output, train_hist],[qmap_train_output, qmap_output, train_hist]]
    names = [['SOM train', 'SOM wet years', 'Train Obs'],['RF train', 'RF wet years', 'Train Obs'],['QMAP train', 'QMAP wet years','Train Obs']]
    i = 0
    letters = ['a)','b)','c)']
    for axis in axes:
        for j in range(len(outputs[i])):
            print(round(np.mean(outputs[i][j]), 2), names[i][j])
        plot_histogram(outputs[i], names[i], test_hist, downscaling_target=downscaling_target, ax=axis, fig=fig)
        axis.text(axis.get_xlim()[0], axis.get_ylim()[1] + 0.002, letters[i])
        i+=1
    plt.savefig('example_figures/ohare_maxYears_trainTest_'+downscaling_target+'_methods_compare_histogram3panel.png')
    plt.show()


def plot_kde(outputs, names, hist_data, scores=None, downscaling_target=None, ax=None, fig=None):
    i = 0
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    for output in outputs:
        if scores is not None:
            sns.kdeplot(output, label=names[i] + ' ' + str(scores[names[i]][0]), ax=ax)
        else:
            sns.kdeplot(output, label=names[i], ax=ax)
        i += 1
    sns.kdeplot(hist_data, color='k', lw=2.0, label='Obs', ax=ax)
    if downscaling_target == 'max_temp':
        plt.xlabel(r'Daily Max Temperature ($^\circ$C)')
    elif downscaling_target == 'precip':
        plt.xlabel(r'PRCP ($10^x$ mm/day)')
    return fig, ax


def plot_histogram(outputs, names, hist_data, fig = None, ax = None, downscaling_target=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    bin_starts = np.array([0, 0.01, .1, .25, .75, 2, 10]) * 25.4
    outputs.append(hist_data)
    names.append('Wet Years Obs')
    ax.hist(outputs, bins=bin_starts, label=names, density=True, rwidth=.4, log=True)
    plt.xscale("log")
    logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)
    ax.xaxis.set_major_formatter(logfmt)
    ax.set_xlabel(r'PRCP ($10^x$ mm/day)')
    ax.yaxis.set_major_formatter(logfmt)
    ax.set_ylabel(r'Frequency ($10^x$)')
    ax.tick_params(axis='both')
    ax.legend()
    return fig, ax


def plot_autocorrelation(outputs, names, hist_data, nlags=10):
    # plot the autocorrelation function for the different downscaling outputs.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    i = 0
    for output in outputs:
        auto_corr = acf(output, nlags=nlags)
        ax.plot(auto_corr, label=names[i])
        i += 1

    obs_corr = acf(hist_data, nlags=nlags)
    ax.plot(obs_corr, label='Obs', color='k')
    ax.set_xlabel('Lag Time (days)')
    ax.set_ylabel('Correlation')
    ax.legend(ncol=2)
    return fig, ax


if __name__ == '__main__':
    downscale_example(downscaling_target='max_temp')
