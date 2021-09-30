#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:51:19 2021

@author: lferiani
"""

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from experiment_constants import MD_COLS, STIM_START_S
from timeseries_helper import plot_stimuli


def plot_frac(df, modecolnames, ax=None, **kwargs):
    """plot_frac
    plots modecolname of df with shaded errorbar
    example:
        plot_frac(frac_motion_mode_with_ci, 'frac_worms_fw', ax=ax)
    """
    if ax is None:
        ax = plt.gca()
    coldict = {'frac_worms_fw': ('tab:green', 'forwards'),
               'frac_worms_bw': ('tab:orange', 'backwards'),
               'frac_worms_st': ('tab:purple', 'stationary'),
               'frac_worms_nan': ('tab:gray', 'undefined')}
    # styledict = {'N2': '--',
    #              'CB4856': '-'}
    for strain, df_g in df.groupby('worm_strain'):
        df_g = df_g.droplevel('worm_strain')
        for col in modecolnames:
            color, motmode = coldict[col]
            df_g.plot(y=col, ax=ax,
                      color=color,
                      # linestyle=styledict[strain],
                      label=strain+' '+motmode,
                      **kwargs)
            lower = df_g[col+'_ci_lower']
            upper = df_g[col+'_ci_upper']
            ax.fill_between(x=lower.index,
                            y1=lower.values,
                            y2=upper.values,
                            alpha=0.3,
                            facecolor=ax.lines[-1].get_color())
    ax.set_ylim((0, 1))
    ax.set_ylabel('fraction of worms')
    return


def plot_stacked_frac_mode(df, strain=None, **kwargs):
    """plot_stacked_frac_mode
    make AoE II style cumulative fraction plot"""

    if ('worm_strain' in df.index.names):
        fig = []
        for strain, df_g in df.groupby('worm_strain'):
            df_g = df_g.droplevel('worm_strain')
            ff = plot_stacked_frac_mode(df_g, strain=strain, **kwargs)
            fig.append(ff)

    else:

        fig, ax = plt.subplots(**kwargs)
        fracalpha = 0.5
        erralpha = 0.5
        facecolours = [(*t, fracalpha) for t in sns.color_palette()]
        coldict = {'frac_worms_fw': ('tab:green', 'forwards'),
                   'frac_worms_bw': ('tab:orange', 'backwards'),
                   'frac_worms_st': ('tab:purple', 'stationary'),
                   'frac_worms_nan': ('tab:gray', 'undefined')}
        yprev = 0
        ycum = 0
        for icol, col in enumerate(['frac_worms_'+x
                                    for x in ['fw', 'st', 'bw', 'nan']]):
            # line between fractions:
            yprev = ycum
            ycum = ycum + df[col]
            ax.fill_between(x=df.index,
                            y1=yprev,
                            y2=ycum,
                            label=col,
                            linewidth=0.5,
                            linestyle='-',
                            facecolor=coldict[col][0],
                            alpha=fracalpha,
                            edgecolor='darkgray')
        # now lower and upper bounds of frac fw
        ylow = df['frac_worms_fw_ci_lower']
        yupp = df['frac_worms_fw_ci_upper']
        ax.fill_between(x=df.index,
                        y1=ylow,
                        y2=yupp,
                        facecolor='gray',
                        edgecolor='None',
                        alpha=erralpha,
                        linewidth=0.2,
                        linestyle='--')
        # get errorbar for frac bw. then sum them to frac_fw+frac_st
        offset = (df['frac_worms_fw']
                  + df['frac_worms_st']
                  - df['frac_worms_bw'])
        ylow = df['frac_worms_bw_ci_lower'] + offset
        yupp = df['frac_worms_bw_ci_upper'] + offset
        # import pdb; pdb.set_trace()
        ax.fill_between(x=df.index,
                        y1=ylow,
                        y2=yupp,
                        facecolor='gray',
                        edgecolor='None',
                        alpha=erralpha,
                        linewidth=0.2,
                        linestyle='--')

        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)
        ax.set_xlim(df.index.min(), df.index.max())
        ax.set_ylabel('cumulative fraction')
        ax.set_title(strain)

        if 'time_s' in df.reset_index().columns:
            plot_stimuli(ax=ax, units='s', stimulus_start=STIM_START)
            ax.set_xlabel('time, (s)')
        else:
            plot_stimuli(ax=ax, units='frames', stimulus_start=STIM_START)
            ax.set_xlabel('time, (frames)')
    return fig


def shadederrorbar(
        x, y, data, which_err='sem',
        fig_kwargs={}, plot_kwargs={}, patches_kwargs={}, leg_kwargs={}):

    if y == 'all':
        y = data.columns.to_list()

    if not isinstance(y, list):
        y = [y]

    df_plot = data.groupby(x)[y].agg(['mean', which_err]).reset_index()
    # import pdb; pdb.set_trace()
    for y_ in y:
        df_plot[(y_, 'lower')] = (df_plot[(y_, 'mean')]
                                  - df_plot[(y_, which_err)])
        df_plot[(y_, 'upper')] = (df_plot[(y_, 'mean')]
                                  + df_plot[(y_, which_err)])
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots(**fig_kwargs)
    for y_ in y:
        df_plot.plot(x, (y_, 'mean'), ax=ax, **plot_kwargs, label=y_)

    patch_colors = [line.get_color() for line in ax.get_lines()]
    for y_, patch_color in zip(y, patch_colors):
        ax.fill_between(x=df_plot[x],
                        y1=df_plot[(y_, 'lower')],
                        y2=df_plot[(y_, 'upper')],
                        color=patch_color,
                        alpha=0.4,
                        **patches_kwargs)
    plt.legend(**leg_kwargs)
    return fig, ax


# def plot_timeseries_multipage(timeseries_df, feats_toplot, figures_dir):
#     for ledint, timeseries_df_g in timeseries_df.groupby('led_intensity'):
#         with PdfPages(
#                 figures_dir / 'downsampled_feats_ledint{}.pdf'.format(ledint),
#                 keep_empty=False) as pdf:
#             for feat in tqdm(feats_toplot):
#                 fig, ax = plt.subplots(figsize=(12.8, 4.8))
#                 sns.lineplot(x='time_binned_s', y=feat,
#                              hue='motion_mode',
#                              style='worm_strain',
#                              data=timeseries_df_g.query('motion_mode != 0'),
#                              estimator=np.mean, ci='sd',
#                              legend='full')
#                 plot_stimuli(ax=ax, units='s',
#                              stimulus_start=STIM_START)
#                 pdf.savefig(fig)
#                 plt.close(fig)


def plot_timeseries_multipage(
        timeseries_data, feats_toplot, fig_savepath,
        time_col='timestamp_binned_s',
        plot_units='s', fps=None,
        stimulus_start_s=[60, 160, 260],
        stimulus_duration_s=10,
        **sns_kwargs,
        ):
    """
    plot_timeseries_multipage Wrapper for sns.lineplot.
    Plot feats in feats_toplot into a multipage pdf at fig_savepath.

    This function does not do any filtering to the data, so make sure
    timeseries_data comes pre-filtered!!!

    Parameters
    ----------
    timeseries_data : Pandas Dataframe
        from /timeseries_data, possibly downsampled.
    feats_toplot : list
        features to plot. Each feature will be in its own plot
    fig_savepath : string or Path
        path to output pdf file
    time_col : str, optional
        which colums of the timeseries_data dataframe to use as x axis,
        by default 'timestamp_binned_s'
    plot_units : str, optional
        's' or 'frames', used by plot_stimuli
    fps : float, optional
        necessary if plot_units is 'frames'
    stimulus_start_s : list, optional
        times at which the blue leds come online, in seconds
    stimulus_duration_s : float, optional
        duration of each burst of blue light, in seconds
    sns_kwargs : optional
        name, value parameters to further pass to seaborn.
        By default this function uses estimator=np.mean, ci='sd', legend='full'
    """

    # default values for sns.lineplot
    default_sns_kwargs = {'estimator': np.mean, 'ci': 'sd', 'legend': 'full'}
    for k, v in default_sns_kwargs.items():
        if k not in sns_kwargs.items():
            sns_kwargs[k] = v

    # open a pdf
    with PdfPages(fig_savepath, keep_empty=False) as pdf:
        # one page per feat to plot
        for feat in tqdm(feats_toplot):
            # create standard size figure
            fig, ax = plt.subplots(figsize=(12.8, 4.8))
            # timeseries plot
            sns.lineplot(
                x=time_col, y=feat, data=timeseries_data, **sns_kwargs
                )
            # add stimuli
            plot_stimuli(
                ax=ax,
                plot_units=plot_units,
                fps=fps,
                stimulus_start_s=stimulus_start_s,
                stimulus_duration_s=stimulus_duration_s)
            pdf.savefig(fig)
            plt.close(fig)

    return


def plot_onehot_variables_timeseries(
        timeseries_df,
        x_col='timestamp_binned_s',
        onehot_value_cols=[],
        xlabel=None,
        ylabel=None,
        title=None,
        ax=None,
        ylim=None,
        xlim=None,
        **sns_kwargs
        ):

    # input checks
    if not isinstance(onehot_value_cols, list):
        onehot_value_cols = [onehot_value_cols]
    for col in onehot_value_cols + ['well_name', 'worm_index']:
        assert col in timeseries_df.columns
    for col in onehot_value_cols:
        assert timeseries_df[col].astype(float).isin([0, 1]).all(), (
            'onehot_values_cols should only be zeros and ones'
        )
    assert 'variable' not in timeseries_df.columns
    assert 'value' not in timeseries_df.columns
    assert 'estimator' in sns_kwargs.keys(), (
        'for safety, explicitly state the estimator')
    assert 'hue' not in sns_kwargs.keys(), ((
        'hue is reserved to be the dummy variable in the long df. '
        'hues will discriminate your onehot_value_cols'
        ))
    # set default ci to None for speed. sns defaults to bootstrap 95th
    if 'ci' not in sns_kwargs.keys():
        sns_kwargs['ci'] = None

    # timseries is a wide dataframe with columns value_cols storing 1,0
    # we need to melt the df to use sns.lineplot properly

    df_long = timeseries_df.melt(
        id_vars=x_col,
        value_vars=onehot_value_cols,
    )
    df_ram_usage = df_long.memory_usage(deep=True).sum() // 1024**2
    print(f'df_long uses {df_ram_usage} MB ram')

    # create figure if we need it
    if ax is None:
        fig, ax = plt.subplots(figsize=(12.8, 4.8))
    else:
        fig = ax.figure

    # actual plot
    plot = sns.lineplot(
        data=df_long,
        x=x_col,
        y='value',
        hue='variable',
        ax=ax,
        **sns_kwargs,
    )

    # make a oneline legend:
    # first remove old legend
    handles, labels = plot.axes.get_legend_handles_labels()
    plot.get_legend().remove()
    # fix limits
    if xlim is None:
        ax.set_xlim((df_long[x_col].min(), df_long[x_col].max()))
    else:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    # re-add oneline legend
    plot.axes.legend(
        handles, labels, ncol=len(onehot_value_cols), loc='upper center',
        bbox_to_anchor=(0.5, 1), frameon=True)

    # put labels on axis now
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return fig, ax
