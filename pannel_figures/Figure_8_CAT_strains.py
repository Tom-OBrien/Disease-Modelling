#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:27:46 2021

Script for making DRD/dopamine defeciancy strains (cat-2 and cat-4) figures

@author: tobrien
"""
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import chain
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions

sys.path.insert(0, '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary')
from helper import (read_disease_data,
                    select_strains,
                    filter_features,
                    make_colormaps,
                    find_window,
                    # long_featmap,
                    BLUELIGHT_WINDOW_DICT,
                    STIMULI_ORDER)
from plotting_helper import  (plot_colormap,
                              plot_cmap_text,
                              feature_box_plots,
                              average_feature_box_plots,
                              window_errorbar_plots,
                              CUSTOM_STYLE,
                              clustered_barcodes)
from ts_helper import (MODECOLNAMES,
                       plot_frac_by_mode,
                       short_plot_frac_by_mode)

# Set paths to files with metadata and experimental results:
# My data:
my_FEAT_FILE =  Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/features_summary_tierpsy_plate_filtered_traj_compiled.csv') 
my_FNAME_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')
my_METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/exploratory_metadata/EXPLORATORY_metadata_with_wells_annotations.csv')
# Ida's data:
ida_FEAT_FILE =  Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/summary_results_files/eleni_filters/features_summary_tierpsy_plate_filtered_traj_compiled.csv') #list(ROOT_DIR.rglob('*filtered/features_summary_tierpsy_plate_20200930_125752.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/eleni_filtered/features_summary_tierpsy_plate_filtered_traj_compiled.csv')# #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/features_summary_tierpsy_plate_20200930_125752.csv')
ida_FNAME_FILE = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/summary_results_files/eleni_filters/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')#list(ROOT_DIR.rglob('*filtered/filenames_summary_tierpsy_plate_20200930_125752.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/eleni_filtered/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')#  #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/filenames_summary_tierpsy_plate_20200930_125752.csv')
ida_METADATA_FILE = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/AuxiliaryFiles/wells_updated_metadata.csv') #list(ROOT_DIR.rglob('*wells_annotated_metadata.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/AuxiliaryFiles/wells_annotated_metadata.csv')
RAW_DATA_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/All_timeseries_hdf5_results')

# Tierpsy window summaries
my_WINDOW_FILES = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/WindowSummaries')
ida_WINDOW_FILES = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/IdasData/IdasWindowResults/window_summaries')
WINDOW_METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/Combined_metadata_and_FeatureMatrix/Combined_DiseaseScreen_metadata.csv')
# Root and save directories
ROOT_DIR = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModelling_Paper_and_Final_Figures/Paper_Figures/Figure_8')

#%% 
# Select example features to be plotted as box plots
CAT_FEATURES = [ 
                'angular_velocity_w_paused_abs_50th_prestim',
                'length_50th_prestim',
                'speed_w_backward_50th_prestim',
                'speed_50th_prestim',
                ]

# Selected blue light 'window' features to make line plots of features across
# prestim, blue light and post-stim conditions to view changes between them
CAT_BLUELIGHT = [ 
                'motion_mode_backward_fraction_bluelight',
                'motion_mode_forward_fraction_bluelight',
                'angular_velocity_midbody_abs_50th_bluelight',
                'speed_50th_bluelight'
                ] 
# Select target and control strains
STRAINS = [
            'cat-2',
            'cat-4',
           ]
CONTROL_STRAIN = 'N2'

#%% Import my data and filter accordingly:
if __name__ == '__main__':
 
    saveto = ROOT_DIR

    # Read in my data
    my_featMat, my_metadata = read_hydra_metadata(my_FEAT_FILE,
                                            my_FNAME_FILE,
                                            my_METADATA_FILE)
    
    # Select only disease model tracking plates and filter out rest of data
    mask = my_metadata['imaging_plate_id'].isin([
        'P01_sh01_01',
        'P01_sh02_01',
        'P01_sh03_01',
        'P01_sh01_02',
        'P01_sh02_02',
        'P01_sh03_02',
        'P01_sh01_03',
        'P01_sh02_03',
        'P01_sh03_03',
        'P02_sh01',
        'P02_sh02',
        'P02_sh03',])
    my_featMat = my_featMat[~mask]
    my_metadata = my_metadata[~mask]
    
    # Remove genes not to include in analysis
    my_genes_to_drop = [
        'Day-2 N2',
        'Day-2 OMG35',
        'Day-4 N2',
        'Day-4 OMG35',
        'glr-1',
        'glr-4',
        'glc-2',
        'OMG35',
        ]
    my_metadata = my_metadata[~my_metadata['worm_gene'].isin(my_genes_to_drop)]
    my_featMat = my_featMat.loc[my_metadata.index]
 
    mask = my_metadata['worm_strain'].isin([
        'Day-2 N2',
        'OMG35'])  
    my_featMat = my_featMat[~mask]
    my_metadata = my_metadata[~mask]
 
     # Filter out columns different from Ida's metadata to merge without issue
    my_columns_to_drop = [
        'column_range', 
        'drug_code', 
        'drug_type', 
        'end_col', 
        'end_column',
        'end_row', 
        'imaging_plate_drug_concentration_uM',
        'imaging_plate_drug_concentration_units', 
        'middle_wormsorter_time',
        'plate_number', 
        'row_range', 
        'solvent', 
        'source_plate_id', 
        'start_col',
        'start_column', 
        'start_row', 
        'stock_drug_conc_M', 
        'wormsorter_end_time',
        'wormsorter_start_time']
    
    my_metadata = my_metadata.drop(my_columns_to_drop,
                                   axis=1)
    
    # my_metadata['date_yyyymmdd'] = pd.to_datetime(
    # my_metadata['date_yyyymmdd'], format='%Y%m%d').dt.date

    # Use glob to find all window summary results files in directory
    my_window_files = list(my_WINDOW_FILES.rglob('*_window_*'))
    my_window_feat_files = [f for f in my_window_files if 'features' in str(f)]
    my_window_feat_files.sort(key=find_window)
    my_window_fname_files = [f for f in my_window_files if 'filenames' in str(f)]
    my_window_fname_files.sort(key=find_window)

    assert (find_window(f[0]) == find_window(f[1]) for f in list(zip(my_window_feat_files, my_window_fname_files)))

#%% Do the same steps as above with Ida's data:
    
    ida_featMat, ida_metadata = read_hydra_metadata(ida_FEAT_FILE,
                                            ida_FNAME_FILE,
                                            ida_METADATA_FILE)    
    ida_genes_to_drop = [
        'cat-4',
        'dys-1',
        'pink-1',
        'gpb-2',
        'kcc-2',
        'snf-11',
        'snn-1',
        'unc-25',
        'unc-43',
        'unc-49',
        'figo-1',
        'C43B7.2'
        ]
    ida_metadata = ida_metadata[~ida_metadata['worm_gene'].isin(
        ida_genes_to_drop)]
    ida_featMat = ida_featMat.loc[ida_metadata.index]
    
    ida_columns_to_drop = [
        'well_label',
        'imaging_run']

    ida_metadata = ida_metadata.drop(ida_columns_to_drop,
                                     axis=1)     
     
    # ida_metadata['date_yyyymmdd'] = pd.to_datetime(
    # ida_metadata['date_yyyymmdd'], format='%Y%m%d').dt.date  
    
    ida_window_files = list(ida_WINDOW_FILES.rglob('*_window_*'))
    ida_window_feat_files = [f for f in ida_window_files if 'features' in str(f)]
    ida_window_feat_files.sort(key=find_window)
    ida_window_fname_files = [f for f in ida_window_files if 'filenames' in str(f)]
    ida_window_fname_files.sort(key=find_window)

    assert (find_window(f[0]) == find_window(f[1]) for f in list(zip(ida_window_feat_files, ida_window_fname_files)))
    
#%% Concatenate metadata and feature matrix dataframes
# TODO: Check size of concat and pd.ignore index = true/false
    meta = pd.concat([my_metadata, ida_metadata], ignore_index=True)
    feat = pd.concat([my_featMat, ida_featMat], ignore_index=True)


    
    # Alight by blue light using tierpsy tools function
    feat, meta = align_bluelight_conditions(feat,
                                            meta,
                                            how='inner') #removes wells that don't have all 3 conditions
    # Drop date where mistake in imaging was made
    date_drop = ['20210212']
    meta = meta[~meta['date_yyyymmdd'].isin(date_drop)]
    feat = feat.loc[meta.index]
    
    # Convert date to 'date_time' and extract date only (for plotting purposes)
    meta['date_yyyymmdd'] = pd.to_datetime(
    meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    
    # Filter out nan's within specified columns and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'imaging_date_yyyymmdd']]

    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)
    
    #%%
    # Make a list of unique genes from combined metadata df that are different
    # to the control and re-name them accordingly
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    genes = [g.replace('C43B7.2', 'figo-1') for g in genes]
    genes = [g for g in genes if 'myo' not in g and 'unc-54' not in g]  


    # Uses Ida's helper function to select candidate genes and select only
    # the data associated with these and the control for analysis
    feat_df, meta_df, idx, gene_list = select_strains(candidate_gene=STRAINS,
                                                    control_strain=CONTROL_STRAIN,
                                                    feat_df=feat,
                                                    meta_df=meta)
    
    # Wrapper function that performs the following:
        # remove wells annotated as bad
        # remove features and wells with too many nans and std=0
        # removes path curvature features
        # makes a feature list of 'all' feature set, i.e. pre/post-stim & bluelight
    feat_df, meta_df, featsets = filter_features(feat_df,
                                                 meta_df)
    
    # Ida's helper function that makes a strain look up table of the selected
    # strains and also makes a unique colour for these (orded by complete 
    # strain list), also makes colour map for the stimuli
    # This function relies on 'idx' calculated by the select strains function
    strain_lut, stim_lut, feat_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=idx,
                                                    candidate_gene=STRAINS
                                                    )

    # Save these colour maps in two different formats
    plot_colormap(strain_lut)
    plt.savefig(saveto / 'strain_cmap.png')
    plot_cmap_text(strain_lut)
    plt.savefig(saveto / 'strain_cmap_text.png')

    plot_colormap(stim_lut, orientation='horizontal')
    plt.savefig(saveto / 'stim_cmap.png')
    plot_cmap_text(stim_lut)
    plt.savefig(saveto / 'stim_cmap_text.png')

    plt.close('all')

    #%% 
    # Make nice box plots of the selected features, including day averages of
    # the selected feature (show_raw_data = false plots no day averages)
    # Note that add stats uses built in stats test, NOT permutation tests
    # TODO: Write function that finds selected stats test from .csv file and adds as custom text
    for f in  CAT_FEATURES:
        average_feature_box_plots(f,
                          feat_df,
                          meta_df,
                          strain_lut,
                          show_raw_data=True,
                          add_stats=False)
        
        plt.savefig(saveto / 'average_boxplots' /'{}_boxplot.png'.format(f),
                    bbox_inches='tight',
                    dpi=200)
    plt.close('all')
    
    #%% plot a heatmap/barcode
    
    # Impute nans with tierpsy tools function
    feat_nonan = impute_nan_inf(feat_df)
    
    # Calculate Z-scores of features and save as data frame
    # featsets is an output of the filter_features function
    featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], axis=0),
                         columns=featsets['all'],
                         index=feat_nonan.index)
    
    # Double check no nan's in the Z-score feature matrix
    assert featZ.isna().sum().sum() == 0    
    
    # N2_clustered_features.txt is a file I have already made during the overall
    # analysis of data. Contains how features are ordered according to z-score
    # this keeps the order of these features consistent across plots
    
    # Find N2clustered feats file 
    N2clustered_features = {}
    for fset in STIMULI_ORDER.keys():
        # N2clustered_features[fset] = []
        with open(ROOT_DIR /  'N2_clustered_features_{}.txt'.format(fset), 'r') as fid:
            N2clustered_features[fset] = [l.rstrip() for l in fid.readlines()]
            # NB: rstrip just removes trailing characters from lines
            
    with open(ROOT_DIR / 'N2_clustered_features_{}.txt'.format('all'), 'r') as fid:
        N2clustered_features['all'] = [l.rstrip() for l in fid.readlines()]

    N2clustered_features_copy = N2clustered_features.copy()
    (saveto / 'heatmaps').mkdir(exist_ok=True)
    
    # !! From here on barcodes for heatmaps not working and I'm unsure what is
    # going on with the helper codes !!
    from plotting_helper import make_heatmap_df, make_barcode, make_clustermaps
    
    for stim,fset in featsets.items():
        heatmap_df = make_heatmap_df(N2clustered_features_copy[stim],
                                     featZ[fset],
                                     meta_df)
        
        make_barcode(heatmap_df,
                     CAT_FEATURES,
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
    
    #%% Make window plots of response to blue light stimuli
    
    # Make 2 empty lists for idas data
    i_feat_windows = []
    i_meta_windows = []
    
    # For loop that iterates over feat anf fname files, obtaining index with
    # enumerate,list these and get the elements of list with zip
    # read_disease data is a wrapper function that:
        # Reads in data with tierpsy tools
        # Aligns by bluelight with tierpsy tools
        # Removes nans with tierpsy tools
    for c,f in enumerate(list(zip(ida_window_feat_files, ida_window_fname_files))):
        i_feat, i_meta = read_disease_data(f[0],
                                         f[1],
                                         ida_METADATA_FILE,
                                         drop_nans=True)
        i_meta['window'] = find_window(f[0])
        
        # Makes a list of data frames of the metadata and feature matrix
        # of each Window summary calculated using Tierpsy (9 [0-8] in total)
        i_meta_windows.append(i_meta)
        i_feat_windows.append(i_feat)

    # Repeat the above steps with my Window summary data
    m_feat_windows = []
    m_meta_windows = []

    for c,f in enumerate(list(zip(my_window_feat_files, my_window_fname_files))):
        m_feat, m_meta = read_disease_data(f[0],
                                         f[1],
                                         my_METADATA_FILE,
                                         drop_nans=True)
        m_meta['window'] = find_window(f[0])
        
        m_meta_windows.append(m_meta)
        m_feat_windows.append(m_feat)

    # Concat metadata
    # Covert list indicies into integers and drop genes as before
    i_meta_windows = pd.concat(i_meta_windows)    
    i_meta_windows = i_meta_windows[~i_meta_windows['worm_gene'].isin(
        ida_genes_to_drop)]
    
    # Do same for my data
    m_meta_windows = pd.concat(m_meta_windows)
    m_meta_windows = m_meta_windows[~m_meta_windows['worm_gene'].isin(
        my_genes_to_drop)]
    mask = m_meta_windows['worm_strain'].isin([
        'Day-2 N2',
        'OMG35'])  
    m_meta_windows = m_meta_windows[~mask]
    
    # concatenate mine and ida's window files together and reset index
    meta_windows = pd.concat([i_meta_windows, m_meta_windows])
    meta_windows.reset_index(drop=True,
                             inplace=True)

    # Extract date only from metadata
    meta_windows['date_yyyymmdd'] = pd.to_datetime(
    meta_windows['date_yyyymmdd'], format='%Y%m%d').dt.date
    
    # Use metadata index to arrange features and concatenate mine and Ida's
    # data into one big DF
    i_feat_windows = pd.concat(i_feat_windows)
    i_feat_windows = i_feat_windows.loc[i_meta_windows.index]
    m_feat_windows = pd.concat(m_feat_windows)
    m_feat_windows = m_feat_windows.loc[m_meta_windows.index]
    feat_windows = pd.concat([i_feat_windows, m_feat_windows])
    feat_windows.reset_index(drop=True,
                             inplace=True)
    
    # Again use the select strains helper function to choose the strains of interest
    feat_windows_df, meta_windows_df, idx, gene_list = select_strains(
                                                candidate_gene=STRAINS,
                                                control_strain=CONTROL_STRAIN,
                                                meta_df=meta_windows,
                                                feat_df=feat_windows)

    # Search for blue_light features only and make df of these only
    bluelight_feats = [f for f in feat_windows_df.columns if 'bluelight' in f]
    feat_windows_df = feat_windows_df.loc[:,bluelight_feats]

    # Same wrapper function as before that performs the following:
        # remove wells annotated as bad
        # remove features and wells with too many nans and std=0
        # removes path curvature features
    feat_windows_df, meta_windows_df, featsets = filter_features(feat_windows_df,
                                                           meta_windows_df)
    
    # Make a list of all the bluelight features 
    bluelight_feats = list(feat_windows_df.columns)
    # Set save path
    (saveto / 'windows_features').mkdir(exist_ok=True)
    
    # Use the hardcoded bluelight window dictionary to append information about
    # light condition, time and stimuli number to the metadata dataframe
    meta_windows_df['light'] = [x[1] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
    meta_windows_df['window_sec'] = [x[0] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]
    meta_windows_df['stim_number'] = [x[2] for x in meta_windows_df['window'].map(BLUELIGHT_WINDOW_DICT)]

    # Group by the stimuli/burst number
    stim_groups = meta_windows_df.groupby('stim_number').groups
    
    #%% Plotting bluelight features
    
    # Iterate over selected features and make a line plot of thesee
    # First plot all 3 stimuli windows
    for f in CAT_BLUELIGHT:
        sns.set_style('ticks')
        (saveto / 'windows_features' / f).mkdir(exist_ok=True)
        window_errorbar_plots(f,
                              feat_windows_df,
                              meta_windows_df,
                              strain_lut,
                              plot_legend=True)
        plt.savefig(saveto / 'windows_features' / f / 'allwindows_{}'.format(f), 
                    bbox_inches='tight', dpi=200)
        plt.close('all')

    # Now plot each window separately based upon stimuli/burst number grouping
        for stim,locs in stim_groups.items():
            window_errorbar_plots(f,
                                  feat_windows_df.loc[locs],
                                  meta_windows_df.loc[locs],
                                  strain_lut,
                                  plot_legend=True)
            plt.savefig(saveto / 'windows_features' / f / 'window{}_{}'.format(stim,f),
                        bbox_inches='tight', dpi=200)
            plt.close('all')
    #%% Timeseries plots
    
    # Make a dictionary of strains (the way I have set this up means I can
    # subset groups of strains together to be plotted on separate plots)
    TS_STRAINS = {'CAT': ['cat-2',
                            'cat-4'
                     ]}  
    
    # Make a list of strains with chain function (returns one iterable
    # from a list of several (not strictly necessary for this set of figs)
    ts_strain_list = list(chain(*TS_STRAINS.values()))
    
    # Find .hdf5 files of selected strains from root directory and read in
    # confidence intervals have already been calculated prior this results
    # in a list of 2 dataframes
    timeseries_df = []
    for g in ts_strain_list:
        _timeseries_fname = RAW_DATA_DIR / '{}_timeseries.hdf5'.format(g)
        timeseries_df.append(pd.read_hdf(_timeseries_fname,
                                          'frac_motion_mode_with_ci'))
  
    # Convert the list into one big dataframe and reset index
    timeseries_df = pd.concat(timeseries_df)
    timeseries_df.reset_index(drop=True, inplace=True)
    # Here I only want to plot the first 160 seconds of timeseries data (only 
    # first pulse of blue light) 
    time_drop = timeseries_df['time_s']>160
    timeseries_df = timeseries_df.loc[~time_drop,:]
 
    # Select all calculated faction modes for strains of interest and control
    frac_motion_modes = [timeseries_df.query('@ts_strain_list in worm_gene')]
    frac_motion_modes.append(timeseries_df.query('@CONTROL_STRAIN in worm_gene').groupby('timestamp').agg(np.mean))
    frac_motion_modes[1]['worm_gene'] = CONTROL_STRAIN
    frac_motion_modes = pd.concat(frac_motion_modes)
    frac_motion_modes.reset_index(drop=True,inplace=True)
    
    # Plot each of the fraction motion modes as separate plots
    # Modecolnames is just hardcoded list of 'fwd, bckwd and stationary' 
    for m in MODECOLNAMES:
        sns.set_style('ticks')
        short_plot_frac_by_mode(frac_motion_modes, strain_lut, modecolname=m)
        if m != 'frac_worms_st':
            plt.ylim([0, 0.5])
        plt.savefig(saveto / 'first_stimuli_ts' /'{}_ts.png'.format(m), dpi=200)
        
            
                        