#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:38:13 2021

@author: lferiani

This should contain generic functions that could go in tierpsytools
"""

import scipy
import warnings
import pandas as pd

from matplotlib import pyplot as plt

from tierpsy.summary.filtering import filter_trajectories
from tierpsy.summary.process_tierpsy import _match_units
from tierpsytools.read_data.get_timeseries import read_timeseries

from experiment_constants import FPS, HIFPS_COLS

TS_NONDATA_COLS = ['worm_index', 'well_name']
MOTION_MODE_NAMES = {-1: 'bw', 0: 'st', 1: 'fw'}
MOTION_MODE_COLS = [f'is_{v}' for v in MOTION_MODE_NAMES.values()] + ['is_nan']
MOTION_CHANGE_COLS = ['bw2fw', 'fw2bw', 'bw2st', 'st2bw', 'st2fw', 'fw2st']


def get_value_from_const_column(df, colname):
    assert df[colname].nunique() == 1, 'Non constant values or all nans!'
    return df[~df[colname].isna()][colname].iloc[0]


def read_and_filter_timeseries_data(
        fname, filter_params, min_thresh_frac_skel=0.8, only_wells=None):
    """
    Reads the timeseries_data for a given file and filter.
    Only read the right wells. if only_wells is none, read all wells
    Returns: timeseries_data
    """
    # TODO: check if I can just read FPS from fname
    cfilter_params, check_ok = _match_units(filter_params, FPS, fname)
    assert check_ok, 'failed units check'

    # read data from file
    timeseries_data = read_timeseries(fname, only_wells=only_wells)
    # filter to keep only worms with > thresh skeletonised worms
    # filter keeps when fun is True - keep if fraction of nonskel is < 0.2
    timeseries_data = timeseries_data.groupby('worm_index').filter(
        lambda x: x['length'].isna().mean() < (1-min_thresh_frac_skel)
        ).reset_index(drop=True)

    # EM: Filter trajectories
    if cfilter_params is not None:
        timeseries_data, _ = \
            filter_trajectories(timeseries_data, None, **cfilter_params)

    if timeseries_data.empty:
        warnings.warn('Ended up with empty timeseries')
        # no data, nothing to do here
        return

    return timeseries_data.reset_index(drop=True)


def downsample_timeseries(
        timeseries_data, fps=25, time_bin_s=1, feat_agg_func='mean'):
    """
    takes timeseries_data dataframe, downsample by aggregating values by
    time_bin_s.
    Aggregating function provided is used on all data columns except for
    status columns (motion mode, on food, turn)
    This is meant to work on single video's timeseries only, as worm_index
    is assumed unique
    """
    # import pdb; pdb.set_trace()
    # convert to seconds, then bin time by flooring to the left of the bin
    timeseries_data['timestamp_s'] = timeseries_data['timestamp'] / fps
    timeseries_data['timestamp_binned_s'] = (
        timeseries_data['timestamp_s'] // time_bin_s * time_bin_s)
    # now time average within each worm. worm_index is unique for the video
    grouping_columns = TS_NONDATA_COLS + ['timestamp_binned_s']

    # functions to use for aggregation.
    agg_funcs = {c: feat_agg_func
                 for c in timeseries_data.columns
                 if c not in grouping_columns}
    for c in ['motion_mode', 'food_region', 'turn']:
        agg_funcs[c] = (lambda x: scipy.stats.mode(x)[0])

    # downsample and apply aggregating functions
    dwnsmpl_df = timeseries_data.groupby(
        grouping_columns, observed=True).agg(agg_funcs).reset_index()

    dwnsmpl_df['well_name'] = dwnsmpl_df['well_name'].astype('category')

    nworms_per_timebin_per_well = calculate_nworm_stats(
        timeseries_data, timebin_column='timestamp_binned_s')

    return dwnsmpl_df, nworms_per_timebin_per_well


def calculate_nworm_stats(
        timeseries_data, timebin_column='timestamp_binned_s'):
    """
    Counts how many worms there are per well per time bin by looking at how
    many different worm_index there are at each timepoint, and taking the
    maximum within the time bin,
    It's important that timeseries_data is not downsampled, as a worm can
    change worm index during a time bin, and this could lead to overestimates
    """
    assert timebin_column in timeseries_data.columns, (
        f'{timebin_column} not in the existing columns'
    )
    # count unique occurrences of well_name and timestamp at each frame
    nworms_per_frame_per_well = (
        timeseries_data[['well_name', timebin_column, 'timestamp']]
        .value_counts()
        .to_frame('n_worms')
        .reset_index()
        )
    # then take the maximum during each time bin
    nworms_per_timebin_per_well = (
        nworms_per_frame_per_well
        .groupby(['well_name', timebin_column], observed=True)['n_worms']
        .max()
        .to_frame(name='n_worms')
        .reset_index()
        .sort_values(by=['well_name', timebin_column])
        .reset_index(drop=True)
        )

    return nworms_per_timebin_per_well


def one_hot_encode_motion_mode(timeseries_data):
    if all([c in timeseries_data.columns for c in MOTION_MODE_COLS]):
        print('motion_mode columns already present, skipping...')
        return timeseries_data
    assert 'motion_mode' in timeseries_data.columns, (
        'no motion_mode column in dataframe')
    timeseries_data['motion_mode_name'] = timeseries_data['motion_mode'].map(
        MOTION_MODE_NAMES)
    timeseries_data = pd.get_dummies(
        timeseries_data,
        columns=['motion_mode_name'],
        prefix='is',
        dummy_na=True,
        )
    return timeseries_data


def find_motion_changes(timeseries_data, time_col='timestamp'):
    """
    This function should be run on one video's worth of data only!
    """
    # checks
    timeseries_data = one_hot_encode_motion_mode(timeseries_data)
    assert 'worm_index' in timeseries_data
    assert time_col in timeseries_data
    assert (
        timeseries_data[['worm_index', time_col]].value_counts().max() == 1
        ), (f'Multiple instances of (worm_index, {time_col}). '
            'This function is for timeseries_data from one video only')
    for wi, wdf in timeseries_data.groupby('worm_index'):
        assert wdf[time_col].is_monotonic_increasing, 'sort timeseries'

    # diff to identify motion change
    timeseries_data['motion_diff'] = timeseries_data.groupby(
        'worm_index')['motion_mode'].transform(
            pd.Series.diff)

    # fill the motion change columns now. watch out for brackets for & and |
    timeseries_data['bw2fw'] = (timeseries_data['motion_diff'] == 2)
    timeseries_data['fw2bw'] = (timeseries_data['motion_diff'] == -2)
    timeseries_data['bw2st'] = (
        (timeseries_data['motion_diff'] == 1) & timeseries_data['is_st'])
    timeseries_data['st2bw'] = (
        (timeseries_data['motion_diff'] == -1) & timeseries_data['is_bw'])
    timeseries_data['st2fw'] = (
        (timeseries_data['motion_diff'] == 1) & timeseries_data['is_fw'])
    timeseries_data['fw2st'] = (
        (timeseries_data['motion_diff'] == -1) & timeseries_data['is_st'])
    timeseries_data['motion_up'] = (timeseries_data['motion_diff'] > 0)
    timeseries_data['motion_down'] = (timeseries_data['motion_diff'] < 0)
    timeseries_data['motion_change'] = (
        timeseries_data['motion_up'] | timeseries_data['motion_down'])
    # check
    assert (
        timeseries_data['motion_change'] ==
        timeseries_data[MOTION_CHANGE_COLS].any(axis=1)
        ).all(), f'motion_change and | of {MOTION_CHANGE_COLS} not matching?'

    return timeseries_data


def plot_stimuli(ax=None, plot_units='s', fps=None,
                 stimulus_start_s=[60, 160, 260],
                 stimulus_duration_s=10):
    """
    plot_stimuli
    plots patches at the times when the stimulus was supposed to be on.
    General function, will go in tierpsytools

    Parameters
    ----------
    ax : figure axis, optional
        a handle to existing axis. default is None, this will create a new
        figure
    plot_units : str, optional
        's' or 'frames', units of the x axis of the plot we're adding the
        stimuli to
    fps : float, optional
        necessary if plot_units is 'frames'
    stimulus_start_s : list, optional
        times at which the blue leds come online, in seconds
    stimulus_duration_s : float, optional
        duration of each burst of blue light, in seconds

    """
    if ax is None:
        ax = plt.gca()

    if plot_units == 'frames':
        assert fps is not None, 'need fps if plot_units is frames'
        stimulus_start = [s * fps for s in stimulus_start_s]
        stimulus_duration = stimulus_duration_s * fps
    else:
        stimulus_start = stimulus_start_s
        stimulus_duration = stimulus_duration_s

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    for ss in stimulus_start:
        rect = plt.Rectangle(xy=(ss, ymin),
                             width=stimulus_duration,
                             height=yrange,
                             alpha=0.2,
                             facecolor=(0, 0, 1))
        ax.add_patch(rect)
    return


def create_bin_edges(window_start, window_width, window_name):
    """
    create_bin_edges
    returns two lists:
        bin_edges = [(left_0, right_0), ..., (left_N, right_N)]
        bin_ids = [(window_name, 0), ..., (window_name, N)]
    where
        right_i - left_i == window_width

    Parameters
    ----------
    window_start : iterable
        left edge of value window
    window_width : scalar, numeric
        width of windows
    window_name : str
        name of the bins
    """
    bin_edges = []
    bin_ids = []
    for c, s in enumerate(window_start):
        bin_ids.append((window_name, c))
        bin_edges.append((s, s+window_width))
    return bin_edges, bin_ids


def aggregate_timeseries_in_bins(
        timeseries_data, bin_edges, bin_ids,
        timecol='timestamp_binned_s', agg_funcs=['mean', 'std']):
    """
    aggregate_timeseries_in_bins
    Two-step:
    - average across all rows that share the same value in the time column
    - aggregate across all rows (so across all timecol values) in the same bin

    Warning: this will average across well names and worm indices!
    It can be called after a groupby (maybe wrapped with partial)

    This is *not* the same as downsampling a worm's trajectory
    (although there might be scope for unifying the two functions)

    Parameters
    ----------
    timeseries_data : pandas df
        DataFrame with timeseries info from Tierpsy.
    bin_edges : iterable of tuples
        specifies left and right edges of time intervals.
        Ideally, created by `create_bin_edges`
    bin_ids : iterable of tuples
        specifies bin type and bin counter for time intervals.
        Ideally, created by `create_bin_edges`
    timecol : str, optional
        column to bin the data by, by default 'timestamp_binned_s'
    err : list of pandas function names, optional
        function to aggregate the values within the same bin.
        By default, ['mean', 'std']

    Returns
    -------
    avg_by_window
        DataFrame with the following columns:
        - 2 id columns (win_type, win_counter),
        - no timecol,
        - len(agg_funcs) x (number of input columns - 1) aggregated features
    """

    # assign window name to timeseries_df rows based on the time intervals
    # https://stackoverflow.com/a/55204612
    binned_timecol = pd.cut(
        timeseries_data[timecol].to_list(),
        pd.IntervalIndex.from_tuples(bin_edges),
        )
    binned_timecol.categories = bin_ids
    timeseries_data['win_id'] = binned_timecol

    # filter away data not in timeseries
    filt_ts_df = timeseries_data[~timeseries_data['win_id'].isna()]
    # split win_id
    filt_ts_df['win_type'], filt_ts_df['win_counter'] = list(
        zip(*filt_ts_df['win_id']))

    # average things by window to get fractions:
    avg_by_window = (
        filt_ts_df
        .groupby(['win_type', 'win_counter', timecol])
        .mean()
        .reset_index()
        .drop(columns=timecol)
        .groupby(['win_type', 'win_counter'])
        .agg(agg_funcs)
    )

    return avg_by_window
