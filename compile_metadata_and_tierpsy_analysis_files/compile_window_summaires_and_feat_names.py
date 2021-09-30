#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:06:18 2021

@author: tobrien

Script for compliling window summaries and feature names for disease model
screen
"""

from pathlib import Path
from tierpsytools.read_data.compile_features_summaries import compile_tierpsy_summaries
import pdb
import sys

sys.path.insert(0, '/Users/tobrien/analysis_repo')


from summarieshelper import (
    concatenate_day_summaries,
    fnamesfeatsmeta_filt_to_full,
    parse_window_number,
    detect_outliers,
    plot_pccontours_2groups_animate,
    )

sum_root_dir = Path('/Volumes/behavgenom$/Tom/Data/Hydra/DiseaseModel/RawData/Results')
list_days = [item for item in sum_root_dir.glob('*/') if item.is_dir() and str(item.stem).startswith('20')]

win_0_fnames_df, win_0_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='0')

win_0_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_0.csv')
win_0_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_0.csv')

win_1_fnames_df, win_1_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='1')

win_1_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_1.csv')
win_1_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_1.csv')

win_2_fnames_df, win_2_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='2')

win_2_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_2.csv')
win_2_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_2.csv')

win_3_fnames_df, win_3_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='3')

win_3_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_3.csv')
win_3_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_3.csv')

win_4_fnames_df, win_4_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='4')

win_4_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_4.csv')
win_4_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_4.csv')

win_5_fnames_df, win_5_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='5')

win_5_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_5.csv')
win_5_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_5.csv')

win_6_fnames_df, win_6_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='6')

win_6_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_6.csv')
win_6_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_6.csv')

win_7_fnames_df, win_7_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='7')

win_7_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_7.csv')
win_7_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_7.csv')

win_8_fnames_df, win_8_feats_df = concatenate_day_summaries(
            sum_root_dir, list_days,
            window='8')

win_8_fnames_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/filenames_summary_tierpsy_plate_filtered_traj_compiled_window_8.csv')
win_8_feats_df.to_csv('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/window_summaries/features_summary_tierpsy_plate_filtered_traj_compiled_window_8.csv')