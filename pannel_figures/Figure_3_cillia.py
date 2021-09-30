#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:23:08 2021

@author: tobrien

Script for plotting the cilliopathy mutants together

"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import chain
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf

sys.path.insert(0, '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary')
from helper import (read_disease_data,
                    select_strains,
                    filter_features,
                    make_colormaps,
                    find_window,
                    # long_featmap,
                    BLUELIGHT_WINDOW_DICT,
                    DATES_TO_DROP,
                    STIMULI_ORDER)
from plotting_helper import  (plot_colormap,
                              plot_cmap_text,
                              feature_box_plots,
                              average_feature_box_plots,
                              window_errorbar_plots,
                              CUSTOM_STYLE)

from ts_helper import (MODECOLNAMES,
                       plot_frac_by_mode,
                       short_plot_frac_by_mode)

ROOT_DIR = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModelling_Paper_and_Final_Figures/Paper_Figures/Figure_3')
FEAT_FILE =  Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/summary_results_files/eleni_filters/features_summary_tierpsy_plate_filtered_traj_compiled.csv') #list(ROOT_DIR.rglob('*filtered/features_summary_tierpsy_plate_20200930_125752.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/eleni_filtered/features_summary_tierpsy_plate_filtered_traj_compiled.csv')# #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/features_summary_tierpsy_plate_20200930_125752.csv')
FNAME_FILE = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/summary_results_files/eleni_filters/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')#list(ROOT_DIR.rglob('*filtered/filenames_summary_tierpsy_plate_20200930_125752.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/eleni_filtered/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')#  #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/filenames_summary_tierpsy_plate_20200930_125752.csv')
METADATA_FILE = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/AuxiliaryFiles/wells_updated_metadata.csv') #list(ROOT_DIR.rglob('*wells_annotated_metadata.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/AuxiliaryFiles/wells_annotated_metadata.csv')

RAW_DATA_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/IdasData/TimeSeries_Results')
WINDOW_FILES = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/IdasData/IdasWindowResults/window_summaries')

CILIA_FEATURES = [
                'length_50th_prestim',
                'width_midbody_norm_50th_prestim',
                'curvature_neck_norm_abs_50th_prestim',
                'd_curvature_midbody_norm_abs_50th_prestim',
                'motion_mode_forward_fraction_prestim'
                    ]

CILIA_BLUELIGHT = ['motion_mode_backward_duration_50th_bluelight',
                   'motion_mode_backward_fraction_bluelight',
                   'motion_mode_backward_frequency_bluelight',
                   'motion_mode_forward_duration_50th_bluelight',
                   'motion_mode_forward_fraction_bluelight',
                   'motion_mode_forward_frequency_bluelight',
                   'speed_midbody_norm_50th_bluelight']

STRAINS = {
            'bb': ['bbs-1',
                  'bbs-2'],
            'other': ['tub-1',
                      'tmem-231']
           }

strain_list = list(chain(*STRAINS.values()))

CONTROL_STRAIN = 'N2'

#%%
if __name__ == '__main__':

    #set style for all figures
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')

    saveto = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModelling_Paper_and_Final_Figures/Paper_Figures/Figure_3')
    saveto.mkdir(exist_ok=True) 
    feat, meta = read_disease_data(FEAT_FILE,
                                   FNAME_FILE,
                                   METADATA_FILE,
                                   export_nan_worms=False)
    
    meta.worm_gene.replace({'C43B7.2':'figo-1'},
                           inplace=True)
    meta['date_yyyymmdd'] = pd.to_datetime(
    meta['date_yyyymmdd'], format='%Y%m%d').dt.date

    window_files = list(WINDOW_FILES.rglob('*_window_*'))
    window_feat_files = [f for f in window_files if 'features' in str(f)]
    window_feat_files.sort(key=find_window)
    window_fname_files = [f for f in window_files if 'filenames' in str(f)]
    window_fname_files.sort(key=find_window)

    assert (find_window(f[0]) == find_window(f[1]) for f in list(zip(window_feat_files, window_fname_files)))

    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    # genes = [g.replace('C43B7.2', 'figo-1') for g in genes]
    genes = [g for g in genes if 'myo' not in g and 'unc-54' not in g]

    #%%
    feat_df, meta_df, idx, gene_list = select_strains(strain_list,
                                                    CONTROL_STRAIN,
                                                    feat_df=feat,
                                                    meta_df=meta)
    # filter features
    feat_df, meta_df, featsets = filter_features(feat_df,
                                                 meta_df)

    # strain_numbers.append(meta_df.groupby('worm_strain')['file_id_prestim'].describe()['count'])


    strain_lut, stim_lut, feat_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=idx,
                                                    candidate_gene=strain_list
                                                    )

    # colorbars to map colors to strains
    plot_colormap(strain_lut)
    plt.savefig(saveto / 'strain_cmap.png')
    plot_cmap_text(strain_lut)
    plt.savefig(saveto / 'strain_cmap_text.png')

    plot_colormap(stim_lut, orientation='horizontal')
    plt.savefig(saveto / 'stim_cmap.png')
    plot_cmap_text(stim_lut)
    plt.savefig(saveto / 'stim_cmap_text.png')

    plt.close('all')

    #%% plot the features
    
    # and make nice plots of the selected figures
    for f in  CILIA_FEATURES:
        average_feature_box_plots(f,
                          feat_df,
                          meta_df,
                          strain_lut,
                          show_raw_data=True,
                          add_stats=False)
        
        plt.savefig(saveto / 'boxplots' /'{}_boxplot.png'.format(f),
                    bbox_inches='tight',
                    dpi=200)
    plt.close('all')
    
    #%% plot a heatmap/barcode
    #zscore
    feat_nonan = impute_nan_inf(feat_df)

    featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], axis=0),
                         columns=featsets['all'],
                         index=feat_nonan.index)

    assert featZ.isna().sum().sum() == 0    
    
    # use the N2 clustered features first
    N2clustered_features = {}
    for fset in STIMULI_ORDER.keys():
        N2clustered_features[fset] = []
        with open(ROOT_DIR /  'N2_clustered_features_{}.txt'.format(fset), 'r') as fid:
            N2clustered_features[fset] = [l.rstrip() for l in fid.readlines()]

    with open(ROOT_DIR / 'N2_clustered_features_{}.txt'.format('all'), 'r') as fid:
        N2clustered_features['all'] = [l.rstrip() for l in fid.readlines()]

    N2clustered_features_copy = N2clustered_features.copy()

    (saveto / 'heatmaps').mkdir(exist_ok=True)
    from plotting_helper import make_heatmap_df, make_barcode, make_clustermaps
    
    for stim,fset in featsets.items():
        heatmap_df = make_heatmap_df(N2clustered_features_copy[stim],
                                     featZ[fset],
                                     meta_df)  
        make_barcode(heatmap_df,
                     CILIA_FEATURES,
                     cm=['inferno']*(heatmap_df.shape[0]-1)+['Pastel1'],
                     vmin_max=[(-2,2)]*(heatmap_df.shape[0]-1)+[(1,3)])

        plt.savefig(saveto / 'heatmaps' / '{}_heatmap.png'.format(stim))
    
    (saveto / 'clustermaps').mkdir(exist_ok=True)
    clustered_features = make_clustermaps(featZ,
                                          meta_df,
                                          featsets,
                                          strain_lut,
                                          feat_lut,
                                           group_vars = ['worm_gene'],
                                          saveto=saveto / 'clustermaps')
     
    #%% plot for the windows
    feat_windows = []
    meta_windows = []
    for c,f in enumerate(list(zip(window_feat_files, window_fname_files))):
        _feat, _meta = read_disease_data(f[0],
                                         f[1],
                                         METADATA_FILE,
                                         drop_nans=True)
        _meta['window'] = find_window(f[0])
        
        meta_windows.append(_meta)
        feat_windows.append(_feat)

    meta_windows = pd.concat(meta_windows)
    meta_windows.reset_index(drop=True,
                             inplace=True)
    meta_windows.worm_gene.replace({'C43B7.2':'figo-1'},
                                   inplace=True)
    
    feat_windows = pd.concat(feat_windows)
    feat_windows.reset_index(drop=True,
                             inplace=True)
    
    feat_windows_df, meta_windows_df, idx, gene_list = select_strains(strain_list,
                                                  CONTROL_STRAIN,
                                                  meta_windows,
                                                  feat_windows)

    # #only need the bluelight features
    bluelight_feats = [f for f in feat_windows_df.columns if 'bluelight' in f]
    feat_windows_df = feat_windows_df.loc[:,bluelight_feats]

    feat_windows_df, meta_windows_df, featsets = filter_features(feat_windows_df,
                                                           meta_windows_df)
    
    bluelight_feats = list(feat_windows_df.columns)


    (saveto / 'windows_features').mkdir(exist_ok=True)
    meta_windows_df['light'] = [x[1] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
    meta_windows_df['window_sec'] = [x[0] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
    meta_windows_df['stim_number'] = [x[2] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]

    stim_groups = meta_windows_df.groupby('stim_number').groups
    
    #%%
    for f in CILIA_BLUELIGHT:
        sns.set_style('ticks')
        (saveto / 'windows_features' / f).mkdir(exist_ok=True)
        window_errorbar_plots(f,
                              feat_windows_df,
                              meta_windows_df,
                              strain_lut,
                              plot_legend=True)
        plt.savefig(saveto / 'windows_features' / f / 'allwindows_{}'.format(f), dpi=200)
        plt.close('all')
        
        for k,v in STRAINS.items():
            window_errorbar_plots(f,
                              feat_windows_df.loc[meta_windows_df.query('@v in worm_gene or @CONTROL_STRAIN in worm_gene').index,:],
                              meta_windows_df.query('@v in worm_gene or @CONTROL_STRAIN in worm_gene'),
                              strain_lut,
                              plot_legend=True)
            plt.savefig(saveto / 'windows_features' / f / 'allwindows_{}_{}'.format(k,f), dpi=200)
            plt.close('all')

        for stim,locs in stim_groups.items():
            window_errorbar_plots(f,
                                  feat_windows_df.loc[locs],
                                  meta_windows_df.loc[locs],
                                  strain_lut,
                                  plot_legend=True)
            plt.savefig(saveto / 'windows_features' / f / 'window{}_{}'.format(stim,f),
                        dpi=200)
            plt.close('all')
            
            for k,v in STRAINS.items():
                window_errorbar_plots(f,
                                      feat_windows_df.loc[set(locs).intersection(set(meta_windows_df.query('@v in worm_gene or @CONTROL_STRAIN in worm_gene').index)),:],
                                      meta_windows_df.loc[set(locs).intersection(set(meta_windows_df.query('@v in worm_gene or @CONTROL_STRAIN in worm_gene').index)),:],
                                      strain_lut,
                                      plot_legend=True)
                plt.savefig(saveto / 'windows_features' / f / 'window{}_{}_{}'.format(stim,k,f),
                            dpi=200)
                plt.close('all')  
            
            
    #%% plot some of the timeseries together as well
    
    timeseries_df = []
    for g in strain_list:
        _timeseries_fname = RAW_DATA_DIR / '{}_timeseries.hdf5'.format(g)
        timeseries_df.append(pd.read_hdf(_timeseries_fname,
                                          'frac_motion_mode_with_ci'))
        
    timeseries_df = pd.concat(timeseries_df)
    timeseries_df.reset_index(drop=True, inplace=True)
    
    frac_motion_modes = [timeseries_df.query('@strain_list in worm_gene')]
    frac_motion_modes.append(timeseries_df.query('@CONTROL_STRAIN in worm_gene').groupby('timestamp').agg(np.mean))
    frac_motion_modes[1]['worm_gene'] = CONTROL_STRAIN
    frac_motion_modes = pd.concat(frac_motion_modes)
    frac_motion_modes.reset_index(drop=True,inplace=True)
    
    for m in MODECOLNAMES:
        sns.set_style('ticks')
        plot_frac_by_mode(frac_motion_modes, strain_lut, modecolname=m)
        if m != 'frac_worms_st':
            plt.ylim([0, 0.5])
        plt.savefig(saveto / '{}_ts.png'.format(m), dpi=200)
        
        for k,v in STRAINS.items():
            plot_frac_by_mode(frac_motion_modes.query('@v in worm_gene or @CONTROL_STRAIN in worm_gene'),
                              strain_lut,
                              modecolname=m)
            if m != 'frac_worms_st':
                plt.ylim([0, 0.5])
            plt.savefig(saveto / '{}_{}_ts.png'.format(k, m), dpi=200)
            
    #%% plot some of the timeseries together as well
    
    timeseries_df = []
    for g in strain_list:
        _timeseries_fname = RAW_DATA_DIR / '{}_timeseries.hdf5'.format(g)
        timeseries_df.append(pd.read_hdf(_timeseries_fname,
                                          'frac_motion_mode_with_ci'))
  
    timeseries_df = pd.concat(timeseries_df)
    timeseries_df.reset_index(drop=True, inplace=True)
    time_drop = timeseries_df['time_s']>160
    timeseries_df = timeseries_df.loc[~time_drop,:]
 
    frac_motion_modes = [timeseries_df.query('@strain_list in worm_gene')]
    frac_motion_modes.append(timeseries_df.query('@CONTROL_STRAIN in worm_gene').groupby('timestamp').agg(np.mean))
    frac_motion_modes[1]['worm_gene'] = CONTROL_STRAIN
    frac_motion_modes = pd.concat(frac_motion_modes)
    frac_motion_modes.reset_index(drop=True,inplace=True)
    
    for m in MODECOLNAMES:
        sns.set_style('ticks')
        short_plot_frac_by_mode(frac_motion_modes, strain_lut, modecolname=m)
        if m != 'frac_worms_st':
            plt.ylim([0, 0.5])
        plt.savefig(saveto / 'first_stimuli_ts' /'{}_ts.png'.format(m), dpi=200)
        
        for k,v in STRAINS.items():
            short_plot_frac_by_mode(frac_motion_modes.query('@v in worm_gene or @CONTROL_STRAIN in worm_gene'),
                              strain_lut,
                              modecolname=m)
            if m != 'frac_worms_st':
                plt.ylim([0, 0.5])
            plt.savefig(saveto / 'first_stimuli_ts' / '{}_{}_ts.png'.format(k, m), dpi=200)

    
    
        
        
    

