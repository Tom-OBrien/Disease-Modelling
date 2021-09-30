#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:27:01 2021

@author: lferiani
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas.api.types import CategoricalDtype


from experiment_constants import FPS, HIFPS_COLS

from timeseries_helper import (
    read_and_filter_timeseries_data, downsample_timeseries,
    find_motion_changes, get_value_from_const_column,
)

WELL_NAMES = [f'{r}{c+1}' for r in 'ABCDEFGH' for c in range(12)]


def collect_timeseries_from_results(
        metadata_df, results_dir, filter_params, hifps_saveto, lofps_saveto,
        ):
    """
    Loop through the metadata, read timeseries from files, save them to disk.
    In the disk, keep data separated by imaging plate.
    This is a rather experiment specific function, because how to save data
    depends on how we want to read it afterwards
    """

    assert all(metadata_df.dtypes == 'category'), 'make metadata categorical'
    assert 'well_id' in metadata_df.columns, (
        'add a `well_id` unique identifier to metadata_df')

    # loop on video, select which wells need to be read
    for gc, (imgstore, md_g) in enumerate(
            tqdm(metadata_df.groupby('imgstore_name'))):
        # what do we want to read?
        feats_fname = results_dir / imgstore / 'metadata_featuresN.hdf5'
        wells_to_read = list(md_g['well_name'])
        # read data
        print('reading data')
        timeseries_data = read_and_filter_timeseries_data(
            feats_fname, filter_params, only_wells=wells_to_read)

        # extract the hifps features for later use
        hifps_timeseries_data = timeseries_data[HIFPS_COLS].copy()

        # downsample the main df
        print('downsampling timeseries')
        timeseries_data, nworms_data = downsample_timeseries(
            timeseries_data, fps=FPS, time_bin_s=2, feat_agg_func='mean')

        # find motion changes
        print('finding motion changes')
        timeseries_data = find_motion_changes(
            timeseries_data, time_col='timestamp_binned_s')
        hifps_timeseries_data = find_motion_changes(
            hifps_timeseries_data, time_col='timestamp')

        # convert frames to seconds in the hifps table too for easier analysis
        hifps_timeseries_data['timestamp_s'] = (
            hifps_timeseries_data['timestamp'] / FPS)
        # import pdb; pdb.set_trace()

        # save to disk.
        # Let's keep the data separated by imaging plate id and timepoint
        print('writing')
        file_mode = 'a' if (gc > 0) else 'w'
        img_plt_id = get_value_from_const_column(md_g, 'imaging_plate_id')
        timepoint_id = get_value_from_const_column(md_g, 'timepoint')

        timeseries_data['well_name'] = timeseries_data['well_name'].astype(
            CategoricalDtype(categories=WELL_NAMES, ordered=False))
        nworms_data['well_name'] = nworms_data['well_name'].astype(
            CategoricalDtype(categories=WELL_NAMES, ordered=False))
        hifps_timeseries_data['well_name'] = (
            hifps_timeseries_data['well_name'].astype(
                CategoricalDtype(categories=WELL_NAMES, ordered=False)))

        # write
        timeseries_data.to_hdf(
            lofps_saveto,
            f'ts_{img_plt_id}_t{timepoint_id}',
            mode=file_mode,
            format='table',
            append=(gc > 0),
            )
        nworms_data.to_hdf(
            lofps_saveto,
            f'nworms_{img_plt_id}_t{timepoint_id}',
            mode='a',
            format='table',
            append=(gc > 0),
            )
        hifps_timeseries_data.to_hdf(
            hifps_saveto,
            f'ts_{img_plt_id}_t{timepoint_id}',
            mode=file_mode,
            format='table',
            append=(gc > 0),
            data_columns=True,
            )
    return

