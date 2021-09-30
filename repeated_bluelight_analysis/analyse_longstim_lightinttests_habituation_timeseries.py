#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:10:49 2020

@author: lferiani
"""
# %%
import sys

import time
import warnings
import numpy as np
import pandas as pd
from pandas import tseries
import seaborn as sns

from tqdm import tqdm
from scipy import stats
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, '/Users/tobrien/analysis_repo/Repeated Bluelight Timeseries Scripts')
from data_helper import collect_timeseries_from_results
from plots_helper import (plot_timeseries_multipage, 
                          plot_onehot_variables_timeseries)
from experiment_constants import (STIM_START_S, 
                                  STIM_DURATION_S, 
                                  FPS,
                                  HIFPS_COLS, 
                                  MD_COLS, 
                                  IMAGINGRUN2TIMEPOINT)
from timeseries_helper import (MOTION_MODE_COLS, 
                               MOTION_CHANGE_COLS, 
                               create_bin_edges,
                               get_value_from_const_column, 
                               plot_stimuli, 
                               aggregate_timeseries_in_bins)


# %% functions


def is_to_abs(
        featname: str, abs_if_matches: list, abs_if_startswith: list) -> bool:
    """
    is_to_abs whether a feature is to take the absolute value of.
    Return True if `featname` matches an entry in `abs_if_matches`, or if
    it starts with one of the strings in `abs_if_startswith`
    """
    is_it = (featname in abs_if_matches) or any(
        featname.startswith(sw) for sw in abs_if_startswith)
    return is_it


def get_paths_for_data_type(hifps_or_lofps, analyis_dir):
    # sanitise input
    assert hifps_or_lofps in ['hifps', 'lofps'], 'must be hifps or lofps'
    if isinstance(analyis_dir, str):
        analyis_dir = Path(analyis_dir)

    timeseries_fname = hd / f'{hifps_or_lofps}_timeseries.hdf5'
    figures_dir = analyis_dir / f'Figures_{hifps_or_lofps}'

    return timeseries_fname, figures_dir


def load_timeseries_from_quickaccess_hdf5(
        quickaccess_fname, imaging_plate_id, timepoint, only_cols=None):
    """
    save a few lines when loading data.
    Return timeseries dataframe, and the name of the df in the hdf5 it was read
    from
    """
    df_name = f'ts_{imaging_plate_id}_t{timepoint}'
    # pandas does not fail if a column in only_cols is missing from the hdf5 :)
    ts_df = pd.read_hdf(quickaccess_fname, '/' + df_name, columns=only_cols)

    return ts_df, df_name


# %% for some reason cell splitting is broken in vscode right now,
# need to do without if name is main :()

# if __name__ == '__main__':
# %% files and directories

# where are the data?
# root_dir = Path('/Volumes/Extreme SSD'
root_dir = Path('/Users/tobrien/Desktop/BlueLight')
aux_root_dir = root_dir / 'AuxiliaryFiles'
res_root_dir = root_dir / 'Results' / 'Results'

# analysis files and folders
hd = Path('/Users/tobrien/Desktop/BlueLight/Analysis')
hd = hd / 'LongStimulation/'

metadata_fname = Path('/Users/tobrien/Desktop/BlueLight/AuxiliaryFiles/wells_updated_metadata.csv')

# here we write the first time (and then read from here for downstream)
hifps_timeseries_fname, _ = get_paths_for_data_type('hifps', hd)
lofps_timeseries_fname, _ = get_paths_for_data_type('lofps', hd)

# do we want to make the plots for lofps or hifps data?
data2plot = 'lofps'  # lofps or hifps
qa_timeseries_fname, figures_dir = get_paths_for_data_type(data2plot, hd)
figures_dir.mkdir(parents=True, exist_ok=True)
time_col = 'timestamp_binned_s' if data2plot is 'lofps' else 'timestamp_s'

# flags
is_reload_timeseries_from_results = True
is_debug_plots = False  # => ci=None in sns, motion mode plots not saved
plt.ioff()

# check data source exists (i.e. hard drive is connected)
if is_reload_timeseries_from_results:
    assert root_dir.exists(), f'Cannot find {str(root_dir)}'

# %%
# metadata

# load the  metadata file from disk
# translate imaging number to timepoint
# create identifier to point to a well at a point in time

metadata_df = pd.read_csv(metadata_fname)
n_samples = metadata_df.shape[0]
bad_well_cols = [col for col in metadata_df.columns if 'is_bad' in col]

bad = metadata_df[bad_well_cols].any(axis=1)

metadata_df = metadata_df.loc[~bad,:]

print('Bad wells removed: ', n_samples - metadata_df.shape[0])

print('Any remaining bad wells?', metadata_df.is_bad_well.unique())

metadata_df['timepoint'] = metadata_df['imaging_run_number'].map(
    IMAGINGRUN2TIMEPOINT)
metadata_df['well_id'] = (
    metadata_df['imaging_plate_id']
    + '_t' + metadata_df['timepoint'].astype(str)
    + '_' + metadata_df['well_name']
)
metadata_df = metadata_df.astype('category')



# %% read timeseries from tierpsy's results files

if is_reload_timeseries_from_results:

    # filtering parameters used by tierpsy[tools]
    filter_params = {
        'min_traj_length': 1, 'time_units': 'seconds',
        'min_distance_traveled': 10, 'distance_units': 'microns',
        'timeseries_names': ['length', 'width_midbody'],
        'min_thresholds': [100, 10],
        'max_thresholds': [3100, 1000],
        'units': ['microns', 'microns'],
    }

    # this uses tierpytools under the hood, reads and saves an hdf5
    collect_timeseries_from_results(
        metadata_df,
        res_root_dir,
        filter_params,
        hifps_saveto=hifps_timeseries_fname,
        lofps_saveto=lofps_timeseries_fname,
        )

# %% analysis of motion mode in timeseries

# for each plate at each time (these are saved as separate entries in hdf5)
# load the plate's timeseries, take the absolute value of some feats
# then plot

# define feats to take the absolute value of
feats_to_abs = {}
feats_to_abs['is'] = [
    'speed',
    'relative_to_body_speed_midbody',
    'd_speed',
    'relative_to_neck_angular_velocity_head_tip',
    ]
feats_to_abs['startswith'] = [
    'path_curvature',
    'angular_velocity',
]

# define feats to plot
feats_toplot = [
    'speed',
    'abs_speed',
    'angular_velocity',
    'abs_angular_velocity',
    'relative_to_body_speed_midbody',
    'abs_relative_to_body_speed_midbody',
    'abs_relative_to_neck_angular_velocity_head_tip',
    'speed_tail_base',
    'length',
    'major_axis',
    'd_speed',
    'head_tail_distance',
    'abs_angular_velocity_neck',
    'abs_angular_velocity_head_base',
    'abs_angular_velocity_hips',
    'abs_angular_velocity_tail_base',
    'abs_angular_velocity_midbody',
    'abs_angular_velocity_head_tip',
    'abs_angular_velocity_tail_tip',
    ]

print('plotting timeseries')
for (imaging_plate_id, timepoint), _ in metadata_df.groupby(
        by=['imaging_plate_id', 'timepoint'], observed=True):
    print(imaging_plate_id, timepoint)

    # load data from hdf5 (hifps only had few cols saved so should be ok)
    timeseries_df, df_name = load_timeseries_from_quickaccess_hdf5(
        qa_timeseries_fname, imaging_plate_id, timepoint)

    # set some columns to be abs
    for feat in timeseries_df.columns:
        if is_to_abs(feat, feats_to_abs['is'], feats_to_abs['startswith']):z
    timeseries_df['abs_'+feat] = timeseries_df[feat].abs()
    
    ts_ram_usage = timeseries_df.memory_usage(deep=True).sum() // 1024**2
    print(f'uses {ts_ram_usage} MB ram')

    # make plots
    if is_debug_plots:
        continue
    plot_timeseries_multipage(
        timeseries_df.query('motion_mode != 0'),
        [f for f in feats_toplot if f in timeseries_df.columns],
        fig_savepath=(figures_dir / df_name).with_suffix('.pdf'),
        time_col=time_col,
        plot_units='s',
        stimulus_start_s=STIM_START_S,
        stimulus_duration_s=STIM_DURATION_S,
        estimator=np.mean,
        ci=(None if is_debug_plots else 'sd'),
        hue='motion_mode',
        )

# drop the last timeseries to free up memory
del(timeseries_df)

# %% motion mode analysis on downsampled timeseries
# takes a surprisingly long amount of time because of bootstrapping

# disable bootstrapping while debugging
ci_method = None if (is_debug_plots or (data2plot is 'hifps')) else 95

# every time i call plot stimuli I use the same arguments, so:
plot_stim_kwargs = {
    'plot_units': 's',
    'stimulus_start_s': STIM_START_S,
    'stimulus_duration_s': STIM_DURATION_S,
    }
columns_to_load = (
    ['worm_index', 'well_name', time_col, 'timestamp'] +
    MOTION_MODE_COLS + MOTION_CHANGE_COLS +
    ['motion_up', 'motion_down', 'motion_change']
)

print('plot motion mode for downsampled timeseries')
for (imaging_plate_id, timepoint), _ in metadata_df.groupby(
        by=['imaging_plate_id', 'timepoint'], observed=True):
#    if imaging_plate_id != 'ip05_bli10_hy05':
#        continue
    print(imaging_plate_id, timepoint)

    # load data from hdf5, we only care about motion mode columns now
    timeseries_df, df_name = load_timeseries_from_quickaccess_hdf5(
        qa_timeseries_fname, imaging_plate_id, timepoint,
        only_cols=columns_to_load)

    ts_ram_usage = timeseries_df.memory_usage(deep=True).sum() // 1024**2
    print(f'uses {ts_ram_usage} MB ram')

    # timseries is a wide dataframe with columns is_fw, is_bw etc
    # value is 1 or 0 if the motion mode matches.
    # to find the fraction of worms in a motion mode at a particular time
    # we just need the average of the motion mode column at that time
    # we can do it with groupby but we don't get a confidence interval
    # so we can leave it to seaborn instead.

    # Using a wrapper for lineplot
    # it first melt timeframes then uses lineplot

    # fraction of worms
    fig, ax = plot_onehot_variables_timeseries(
        timeseries_df,
        x_col=time_col,
        onehot_value_cols=MOTION_MODE_COLS,
        xlabel='time, (s)',
        ylabel='fraction of worms',
        hue_order=['_', 'is_st', 'is_fw', 'is_nan', 'is_bw'],
        estimator='mean',
        ci=None,
        ylim=(-0.01, 1.01),
        )
    plot_stimuli(ax=ax, **plot_stim_kwargs)
    plt.show()
    fig_fname = f'frac_worms_allmm_{imaging_plate_id}_t{timepoint}.pdf'
    if not is_debug_plots:
        fig.savefig(figures_dir / fig_fname, bbox_inches='tight')
    plt.close(fig)

    # working plot showing the number of worms over time, by motion mode
    fig, ax = plot_onehot_variables_timeseries(
        timeseries_df,
        x_col=time_col,
        onehot_value_cols=MOTION_MODE_COLS,
        xlabel='time, (s)',
        ylabel='number of worms',
        hue_order=['_', 'is_bw', 'is_fw', 'is_nan', 'is_st'],
        estimator='sum',
        ci=None,
        )
    plot_stimuli(ax=ax, **plot_stim_kwargs)
    plt.show()
    fig_fname = f'n_worms_allmm_{imaging_plate_id}_t{timepoint}.pdf'
    if not is_debug_plots:
        fig.savefig(figures_dir / fig_fname, bbox_inches='tight')
    plt.close(fig)

    # working plot showing the total number of worms over time
    assert not timeseries_df['worm_index'].isna().any(), (
        'there shouldnt be any nan in worm index')

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    timeseries_df.groupby(time_col)['worm_index'].count().plot(ax=ax)
    ax.set_xlabel('time, (s)')
    ax.set_ylabel('number of worms')
    plot_stimuli(ax=ax, **plot_stim_kwargs)
    ax.set_xlim(
        timeseries_df[time_col].apply(['min', 'max']).values)
    fig.tight_layout()

    fig_fname = f'n_worms_{imaging_plate_id}_t{timepoint}.pdf'
    if not is_debug_plots:
        fig.savefig(figures_dir / fig_fname, bbox_inches='tight')

    # pretty plots with forward and stationary in different panels
    avg_prestim = timeseries_df[
        timeseries_df[time_col] < STIM_START_S[0]
        ][['is_fw', 'is_st']].mean()

    avg_poststim = timeseries_df[
        timeseries_df[time_col] > 2300
        ][['is_fw', 'is_st']].mean()

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 4.8))

    plot_onehot_variables_timeseries(
        timeseries_df.rename(columns={'is_fw': 'forwards'}),
        x_col=time_col,
        onehot_value_cols=['forwards'],
        xlabel='time, (s)',
        ylabel='fraction of worms',
        hue_order=[None] * 2 + ['forwards'],
        estimator='mean',
        ci=ci_method,
        ax=axs[0],
    )
    axs[0].set_xticklabels([])
    axs[0].set_xlabel(None)
    axs[0].axhline(y=avg_prestim['is_fw'], linestyle='--', color='r')
    axs[0].axhline(y=avg_poststim['is_fw'], linestyle='--', color='r')
    plot_onehot_variables_timeseries(
        timeseries_df.rename(columns={'is_st': 'stationary'}),
        x_col=time_col,
        onehot_value_cols=['stationary'],
        xlabel='time, (s)',
        ylabel='fraction of worms',
        hue_order=[None] * 4 + ['stationary'],
        estimator='mean',
        ci=ci_method,
        ax=axs[1],
    )
    axs[1].axhline(y=avg_prestim['is_st'], linestyle='--', color='r')
    axs[1].axhline(y=avg_poststim['is_st'], linestyle='--', color='r')
    for ax in axs:
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim(
            timeseries_df[time_col].apply(
                ['min', 'max']).values
            )
        plot_stimuli(ax=ax, **plot_stim_kwargs)

    plt.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)  # yes, again
    plt.show()
    fig_fname = f'frac_worms_pretty_{imaging_plate_id}_t{timepoint}.pdf'
    if not is_debug_plots:
        fig.savefig(figures_dir / fig_fname, bbox_inches='tight')
    plt.close(fig)

del(timeseries_df)  # this is ugly maybe a function is needed
print('Done')

# %%
# I want to try and plot the fraction of motion mode of all the reps
# with same stimuli types on the same axis
print('plot motion modes over time in same plot')
for stimtype, md_li in metadata_df.groupby(by='stimuli_type'):   #TODO: Fix stim type, make unique

    # In this experiment we only expect one plate per led intensity:
    #imaging_plate_id = get_value_from_const_column(
    #    md_li, 'imaging_plate_id')
    #print(imaging_plate_id)

    # create a dataframe with the three videos concatenated.
    # the column "timepoint" keeps them separated
    timeseries_df = []
    for tp in sorted(md_li['timepoint'].unique()):
        ts_df, _ = load_timeseries_from_quickaccess_hdf5(
            qa_timeseries_fname, imaging_plate_id, tp,
            only_cols=columns_to_load)
        # add column with timepoint
        ts_df['timepoint'] = f'{tp*2}h'
        ts_df['timepoint'] = ts_df['timepoint'].astype('category')
        timeseries_df.append(ts_df)
    del(ts_df)

    timeseries_df = pd.concat(timeseries_df, axis=0, ignore_index=True)
    ts_ram_usage = timeseries_df.memory_usage(deep=True).sum() // 1024**2
    print(f'uses {ts_ram_usage} MB ram')

    # pretty plots with forward and stationary in different panels
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 9.6))
    sns.lineplot(
        data=timeseries_df.rename(columns={'is_fw': 'forwards'}),
        x=time_col,
        y='forwards',
        hue='timepoint',
        palette='Greens_r',
        estimator='mean',
        ci=ci_method,
        ax=axs[0]
    )
    axs[0].set_xlabel('time, (s)')
    axs[0].set_ylabel('fraction of worms')
    axs[0].set_xticklabels([])
    axs[0].set_xlabel(None)
    sns.lineplot(
        data=timeseries_df.rename(columns={'is_st': 'stationary'}),
        x=time_col,
        y='stationary',
        hue='timepoint',
        palette='Purples_r',
        estimator='mean',
        ci=ci_method,
        ax=axs[1]
    )
    axs[1].set_xlabel('time, (s)')
    axs[1].set_ylabel('fraction of worms')
    for ax in axs:
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim(
            timeseries_df[time_col].apply(
                ['min', 'max']).values
            )
        plot_stimuli(ax=ax, **plot_stim_kwargs)

    plt.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    fig_fname = f'frac_worms_pretty_{imaging_plate_id}_allt.pdf'
    if not is_debug_plots:
        fig.savefig(figures_dir / fig_fname, bbox_inches='tight')
    plt.close(fig)

del(timeseries_df)
print('Done')

# %% pre/post burst investigation

# create lists with time intervals for pre/post burst investigation
# careful the windows are defined as left+width
win_len_s = 10
pre_win_edges_s, pre_win_ids = create_bin_edges(
    STIM_START_S-win_len_s, win_len_s, 'pre')
post_win_edges_s, post_win_ids = create_bin_edges(
    STIM_START_S+STIM_DURATION_S, win_len_s, 'post')
# put them together
win_edges_s = pre_win_edges_s + post_win_edges_s
win_ids = pre_win_ids + post_win_ids

# what to plot, error to use
mm_to_plot = ['is_fw', 'is_st']
err = 'std'

# make this analysis separately for now
for (imaging_plate_id, timepoint), _ in metadata_df.groupby(
        by=['imaging_plate_id', 'timepoint'], observed=True):
    print(imaging_plate_id, timepoint)

    # load data from hdf5
    timeseries_df, df_name = load_timeseries_from_quickaccess_hdf5(
        qa_timeseries_fname, imaging_plate_id, timepoint,
        (['worm_index', 'well_name', time_col] + mm_to_plot)
    )

    avg_by_window = aggregate_timeseries_in_bins(
        timeseries_df[[time_col] + mm_to_plot],
        win_edges_s, win_ids,
        timecol=time_col,
        agg_funcs=['mean', err])

    # split mean and error for plotting purposes
    delta_mean = (
        (avg_by_window.loc['post', [(c, 'mean') for c in mm_to_plot]]
            - avg_by_window.loc['pre', [(c, 'mean') for c in mm_to_plot]])
        .droplevel(level=1, axis=1)
        .rename(columns={'is_fw': 'forwards', 'is_st': 'stationary'})
    )
    delta_err = (
        np.hypot(
            avg_by_window.loc['post', [(c, err) for c in mm_to_plot]],
            avg_by_window.loc['pre', [(c, err) for c in mm_to_plot]])
        .droplevel(level=1, axis=1)
        .rename(columns={'is_fw': 'forwards', 'is_st': 'stationary'})
    )

    # plot
    fig, ax = plt.subplots()
    styledict = dict(marker='o', ls='None', lw=2, capthick=2, capsize=2)
    delta_mean.plot.line(
        yerr=delta_err,
        color=['tab:green', 'tab:purple'],
        ax=ax,
        xlabel='burst number',
        ylabel=r'$\Delta$ fraction of worms',
        xticks=delta_mean.index,
        **styledict
    )
    plt.plot(
        delta_mean.index,
        np.polyval(np.polyfit(
            delta_mean.index, delta_mean['forwards'], 1,
            w=(1/delta_err['forwards'])), x=delta_mean.index),
        color='tab:green')
    plt.plot(
        delta_mean.index,
        np.polyval(np.polyfit(
            delta_mean.index, delta_mean['stationary'], 1,
            w=(1/delta_err['stationary'])), x=delta_mean.index),
        color='tab:purple')
    ax.set_xticklabels(delta_mean.index + 1)
    fig.tight_layout()
    plt.show()

    fig_fname = (
        f'delta_frac_worms_vs_burst_{imaging_plate_id}_t{timepoint}.pdf')
    # if not is_debug_plots:
    fig.savefig(figures_dir / fig_fname, bbox_inches='tight')
    plt.close(fig)

# %% essentially do the same thing again, but this time only two time windows
# before the first burst and after the last one

# time windows. only two, specify by hand is faster
win_edges_s = [(0, 300), (2300, 2600)]
win_ids = [('pre', 0), ('post', 0)]
mm_to_plot = ['is_fw', 'is_st']
err = 'std'

# TODO: join this loop with previous
for (imaging_plate_id, timepoint), _ in metadata_df.groupby(
        by=['imaging_plate_id', 'timepoint'], observed=True):
    print(imaging_plate_id, timepoint)

    # load data from hdf5
    # load data from hdf5
    timeseries_df, df_name = load_timeseries_from_quickaccess_hdf5(
        qa_timeseries_fname, imaging_plate_id, timepoint,
        (['worm_index', 'well_name', time_col] + mm_to_plot)
    )

    avg_by_window = aggregate_timeseries_in_bins(
        timeseries_df[[time_col] + mm_to_plot],
        win_edges_s, win_ids,
        timecol=time_col,
        agg_funcs=['mean', err])

    # split mean and error for plotting purposes
    delta_mean = (
        (avg_by_window.loc['post', [(c, 'mean') for c in mm_to_plot]]
            - avg_by_window.loc['pre', [(c, 'mean') for c in mm_to_plot]])
        .droplevel(level=1, axis=1)
        .rename(columns={'is_fw': 'forwards', 'is_st': 'stationary'})
    )
    delta_err = (
        np.hypot(
            avg_by_window.loc['post', [(c, err) for c in mm_to_plot]],
            avg_by_window.loc['pre', [(c, err) for c in mm_to_plot]])
        .droplevel(level=1, axis=1)
        .rename(columns={'is_fw': 'forwards', 'is_st': 'stationary'})
    )

    # plot
    # massage the data to quickly plot them in different colors and xticks
    delta_mean.loc[1] = delta_mean.loc[0].copy()
    delta_err.loc[1] = delta_err.loc[0].copy()
    delta_mean.loc[0, 'stationary'] = np.nan
    delta_mean.loc[1, 'forwards'] = np.nan
    delta_err.loc[0, 'stationary'] = np.nan
    delta_err.loc[1, 'forwards'] = np.nan

    fig, ax = plt.subplots(figsize=(3.2, 4.8))
    styledict = dict(marker='o', ls='None', lw=2, capthick=2, capsize=2)
    delta_mean.plot.line(
        yerr=delta_err,
        color=['tab:green', 'tab:purple'],
        ax=ax,
        ylabel=r'$\Delta$ fraction of worms across all bursts',
        xlabel='',
        legend=False,
        **styledict
    )
    ax.set_ylim((-0.5, 0.59))
    ax.set_xlim((-0.5, 1.5))
    plt.xticks(
        delta_mean.index, ['forwards', 'stationary'], rotation=45)

    plt.show()
    fig_savepath = f'tired_worms_{imaging_plate_id}_t{timepoint}.pdf'
    if not is_debug_plots:
        fig.savefig(figures_dir / fig_savepath)
    plt.close(fig)
