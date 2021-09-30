#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:02:53 2021

Merge my+Ida's feature summaries, filenames and metadata and filtering data to 
generate cluster maps and PCA plots and example figures for paper.

@author: tobrien
"""
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from scipy import stats
from tierpsytools.read_data.hydra_metadata import (read_hydra_metadata,
                                                   align_bluelight_conditions)
from sklearn.decomposition import PCA
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf


sys.path.insert(0, '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary')
from helper import (filter_features,
                    make_colormaps,
                    select_strains,
                    STIMULI_ORDER)
from plotting_helper import (CUSTOM_STYLE,
                             make_clustermaps,
                             plot_colormap,
                             plot_cmap_text,
                             feature_box_plots,
                             # example_feature_box_plots,
                             make_paper_clustermaps)
from strain_cmap import full_strain_cmap as STRAIN_cmap

#%%

my_FEAT_FILE =  Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/features_summary_tierpsy_plate_filtered_traj_compiled.csv') 
my_FNAME_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')
my_METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/exploratory_metadata/EXPLORATORY_metadata_with_wells_annotations.csv')

ida_FEAT_FILE =  Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/summary_results_files/eleni_filters/features_summary_tierpsy_plate_filtered_traj_compiled.csv') #list(ROOT_DIR.rglob('*filtered/features_summary_tierpsy_plate_20200930_125752.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/eleni_filtered/features_summary_tierpsy_plate_filtered_traj_compiled.csv')# #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/features_summary_tierpsy_plate_20200930_125752.csv')
ida_FNAME_FILE = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/summary_results_files/eleni_filters/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')#list(ROOT_DIR.rglob('*filtered/filenames_summary_tierpsy_plate_20200930_125752.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/eleni_filtered/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')#  #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/filenames_summary_tierpsy_plate_20200930_125752.csv')
ida_METADATA_FILE = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/AuxiliaryFiles/wells_updated_metadata.csv') #list(ROOT_DIR.rglob('*wells_annotated_metadata.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/AuxiliaryFiles/wells_annotated_metadata.csv')

# SAVE_DIR = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModelling_Paper_and_Final_Figures/combined_metadata_and_feature_matrix')
# ANALYSIS_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results')
Paper_Figure_Save_Dir = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModelling_Paper_and_Final_Figures/Paper_Figures')
STRAIN_STATS_DIR = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModelling_Paper_and_Final_Figures/t-test_permutation_strain_figures')
# LMM_STATS_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Final_Paper_Results_and_Figures/LMM_strain_figures')

saveto = Paper_Figure_Save_Dir / 'Figure_1'
# saveto = ANALYSIS_DIR / 'Figures' / 'summary_figures'
saveto.mkdir(exist_ok=True)

CONTROL_STRAIN = 'N2'
EXAMPLES = {
            'curvature_head_norm_abs_50th_prestim': ['bbs-1',
                                                      'avr-14'],
               'speed_midbody_norm_50th_prestim': [
                                                   'glr-1',
                                                   'unc-80',
                                                   ],
               'relative_to_body_radial_velocity_head_tip_w_forward_50th_poststim': [
                                                                                    'glc-2',
                                                                                    'unc-80',
                                                                                    # 'glr-1',
                                                                                    # 'tmem-231',
                                                                                    # 'add-1',
                                                                                    # 'dys-1',
                                                                                    # 'snf-11',
                                                                                    # 'unc-25',
                                                                                    # 'unc-43',
                                                                                    # 'unc-49',
                                                                                    # 'unc-77',
                                                                                    # 'cat-2',
                                                                                    ],
               'length_50th_prestim': [
                                       'tub-1', 
                                        'bbs-1', 
              #                         # 'cat-4', 
              #                         # 'unc-25', 
              #                         # 'figo-1', 
              #                         # 'unc-49', 
              #                         # 'avr-14', 
              #                         # 'bbs-2'
                                       ],
               'width_midbody_norm_50th_prestim': [
                                              'tub-1', 
                                              'bbs-1', 
              #                                # 'bbs-2', 
              #                                # 'figo-1', 
              #                                # 'add-1'
                                              ] 
             }

   #%% Import my data and filter accordingly:
if __name__ == '__main__':
 
    # Read in data
    my_featMat, my_metadata = read_hydra_metadata(my_FEAT_FILE,
                                            my_FNAME_FILE,
                                            my_METADATA_FILE)
    
    my_featMat, my_metadata = align_bluelight_conditions(
                                my_featMat, my_metadata, how='inner')

    #Filter out nan worms and print .csv of these    
    my_nan_worms = my_metadata[my_metadata.worm_gene.isna()][[
                                            'featuresN_filename',
                                            'well_name',
                                            'imaging_plate_id',
                                            'instrument_name',
                                            'date_yyyymmdd']]

    my_nan_worms.to_csv(
        my_METADATA_FILE.parent / 'my_nan_worms.csv', index=False)
    print('{} nan worms - my data'.format(my_nan_worms.shape[0]))
    my_featMat = my_featMat.drop(index=my_nan_worms.index)
    my_metadata = my_metadata.drop(index=my_nan_worms.index)

    #Rename genes for paper
    my_metadata.worm_gene.replace({'C43B7.2':'figo-1'},  
                           inplace=True)    
    # Select on disease model tracking plates
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
    
    # Remove genes to not include
    my_genes_to_drop = [
        'Day-2 N2',
        'Day-2 OMG35',
        'Day-4 N2',
        'Day-4 OMG35',
        'glr-1',
        'glr-4',
        'glc-2',
        'OMG35'
        ]
    my_metadata = my_metadata[~my_metadata['worm_gene'].isin(my_genes_to_drop)]
    my_featMat = my_featMat.loc[my_metadata.index]
 
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
    
    # Appened new column to fit with functions later on
    imaging_date_yyyymmdd = my_metadata['date_yyyymmdd']
    imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
    my_metadata['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd
    my_metadata = pd.DataFrame(my_metadata)    

    # filter features
    my_feat_df, my_meta_df, my_featsets = filter_features(my_featMat,
                                                 my_metadata)

    #%% Import Ida's data and filter accordingly:
    
    ida_featMat, ida_metadata = read_hydra_metadata(ida_FEAT_FILE,
                                            ida_FNAME_FILE,
                                            ida_METADATA_FILE)    

    ida_featMat, ida_metadata = align_bluelight_conditions(
                                    ida_featMat, ida_metadata, how='inner')

    ida_nan_worms = ida_metadata[ida_metadata.worm_gene.isna()][[
                                            'featuresN_filename',
                                            'well_name',
                                            'imaging_plate_id',
                                            'instrument_name',
                                            'imaging_date_yyyymmdd']]

    ida_nan_worms.to_csv(
        ida_METADATA_FILE.parent / 'ida_nan_worms.csv', index=False)
    print('{} nan worms - Idas data'.format(ida_nan_worms.shape[0]))
    ida_featMat = ida_featMat.drop(index=ida_nan_worms.index)
    ida_metadata = ida_metadata.drop(index=ida_nan_worms.index)
    
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
        'C43B7.2',
        # 'N2'
        ]
    ida_metadata = ida_metadata[~ida_metadata['worm_gene'].isin(
        ida_genes_to_drop)]
    ida_featMat = ida_featMat.loc[ida_metadata.index]
    
    ida_columns_to_drop = [
        'well_label',
        'imaging_run']

    ida_metadata = ida_metadata.drop(ida_columns_to_drop,
                                     axis=1)

    for g in ida_metadata['worm_gene'].unique():
        if g.startswith('myo') or g.startswith('unc-54'):
            ida_genes_to_drop.append(g)
    ida_metadata = ida_metadata[~ida_metadata['worm_gene'].isin(ida_genes_to_drop)]
    ida_featMat = ida_featMat.loc[ida_metadata.index]     

    # filter features
    ida_feat_df, ida_meta_df, ida_featsets = filter_features(ida_featMat,
                                                 ida_metadata)
    
    #%% Concatenate my/Ida's dataframes and featuresets
    
    append_feat_df = [my_feat_df, ida_feat_df]
    append_meta_df = [my_meta_df, ida_meta_df]
    
    feat = pd.concat(append_feat_df,
                     axis=0,
                     ignore_index=True)
    meta = pd.concat(append_meta_df,
                     axis=0,
                     ignore_index=True)
    
    feat = pd.DataFrame(feat)
    meta = pd.DataFrame(meta)
    
    feat_df = feat 
    meta_df = meta 
    featsets = ida_featsets
    
    # meta['date_yyyymmdd'] = meta['date_yyyymmdd'].dt.date
    # meta['imaging_date_yyyymmdd'] = meta['imaging_date_yyyymmdd'].dt.date      
    #%%
    #set style for all figures
    plt.style.use(CUSTOM_STYLE)
    sns.set_style('ticks')

    #make colormaps
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
    genes.sort()
    
    # select only neuro disease models
    meta = meta.query('@genes in worm_gene or @CONTROL_STRAIN in worm_gene')
    feat = feat.loc[meta.index,:]
    
    # Make colour maps for strains and stimuli
    strain_lut, stim_lut, feat_lut = make_colormaps(genes,
                                                    featlist=featsets['all']
                                                    )
    # strain_lut=STRAIN_cmap
    plot_cmap_text(strain_lut, 70)
    plt.savefig(saveto / 'strain_cmap.png', bbox_inches="tight", dpi=300)
    plot_colormap(stim_lut, orientation='horizontal')
    plt.savefig(saveto / 'stim_cmap.png', bbox_inches="tight", dpi=300)
    plot_cmap_text(stim_lut)
    plt.savefig(saveto / 'stim_cmap_text.png', bbox_inches="tight", dpi=300)
    plt.close('all')
    
    # impute nans and inf and z score  
    feat_nonan = impute_nan_inf(feat_df)

    featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], axis=0),
                         columns=featsets['all'],
                         index=feat_nonan.index)

    assert featZ.isna().sum().sum() == 0
    #%%
    # Look for strain_stats.csv files in defined directory
    strain_stats = [s for s in (STRAIN_STATS_DIR).glob('**/*_stats.csv')]
    print(('Collating pValues for {} worm strains').format(len(strain_stats)))
    
    # Combine all strain stats into one dataframe and reorder columns so worm
    # gene is the first one (easier to read/double check in variable explorer)
    combined_strains_stats = pd.concat([pd.read_csv(f) for f in strain_stats])
    heat_map_row_colors = combined_strains_stats['worm_gene'].map(strain_lut)
    first_column = combined_strains_stats.pop('worm_gene')
    combined_strains_stats.insert(0, 'worm_gene', first_column )
   
    # Save combined stats as a .csv in save directory:
    # combined_strains_stats.to_csv(saveto / "permutation_combined_strain_stats.csv", index=False)
    
    # Set worm gene as index for dataframe- this removes this comlumn from df
    combined_strains_stats = combined_strains_stats.set_index(['worm_gene'])
    
    # Now count total features in df (total features) and save as variable
    total_feats = len(combined_strains_stats.columns)
    
    # Count nan's in df for each strain/row
    null_feats = combined_strains_stats.isnull().sum(axis=1)
    
    # Compute total number of significant feats for each strain
    sig_feats = total_feats - null_feats
    
    # Save as a dataframe (indexed by worm gene)
    sig_feats = pd.DataFrame(sig_feats)
    
    # Naming column containing number of significant feats
    sig_feats = sig_feats.rename(columns={0: 'Total_Significant_Features'}) 
    
    # Sorting dataframe from most -> fewest significant feats
    sig_feats = sig_feats.sort_values(by='Total_Significant_Features', axis=0, 
                                      ascending=False)
    # Resting index on ordered df for purposes of plotting later on
    sig_feats = sig_feats.reset_index()
    
    # Using sorted index of total sig feats to reorder total stats df
    combined_strains_stats = combined_strains_stats.reindex(index=sig_feats['worm_gene'])
    
    # Rest index (if needed):
    # combined_strains_stats.reset_index(inplace=True)
    print('Total number of features {}'.format(total_feats))
    print(sig_feats)
    
    #%% Make a line plot of total significant features ordered sae as heatmap
    sns.set_style('ticks')
    l = sns.lineplot(data=sig_feats, 
                     x='worm_gene', 
                     y='Total_Significant_Features',
                     color='black')
    plt.xticks(rotation=90, fontsize=13)
    plt.yticks(rotation=45, fontsize=14)
    l.set_xlabel(' ', fontsize=18)
    l.set_ylabel('Number of Significant Features', fontsize=16, rotation=90)
    plt.yticks([0 ,1000, 2000, 3000, 4000, 5000, 6000, 7000], fontsize=14)
    l.invert_yaxis()
    l.axhline(y=0, color='black', linestyle=':', alpha=0.2)
    plt.savefig(saveto / 'sig_feats_lineplot.png', bbox_inches='tight',
                dpi=300)
    
    #%% make clustermap of all strains grouped by worm_gene
    # Note: Is ordered using seaborn distance matrix not by number of sig feats
    save_clustermap = saveto / 'all_strains_clustermaps'

    heatmap_strain_cmap =  {
                    # 'N2': (0.6, 0.6, 0.6), 
                    'add-1': (0.93592664, 0.94467945, 0.84470875), 
                    'avr-14': (0.93982535, 0.90866897, 0.7964528), 
                    'bbs-1': (0.94962434, 0.86199684, 0.76700144), 
                    'bbs-2': (0.95486878, 0.80573384, 0.75808215), 
                    'cat-2': (0.94567816, 0.74297858, 0.76722388), 
                    'cat-4': (0.91457635, 0.67826408, 0.78825288), 
                    'dys-1': (0.85785758, 0.61672956, 0.81242076), 
                    'figo-1': (0.77627444, 0.5632024 , 0.82995768), 
                    'glc-2': (0.67494941, 0.52134581, 0.83178109), 
                    'glr-1': (0.56254037, 0.49301285, 0.81107205), 
                    'glr-4': (0.44981073, 0.47791147, 0.76445837), 
                    'gpb-2': (0.34784927, 0.47363229, 0.69261263), 
                    'kcc-2': (0.26623878, 0.47603156, 0.60017055), 
                    'mpz-1': (0.21147789, 0.47990349, 0.49498741), 
                    'nca-2': (0.18591761, 0.47982939, 0.38685793), 
                    'pink-1': (0.18738894, 0.47106247, 0.28591286), 
                    'snf-11': (0.20958494, 0.45030085, 0.20095838), 
                    'snn-1': (0.24313706, 0.41621964, 0.13803532), 
                    'tmem-231': (0.27721054, 0.36967075, 0.09944094), 
                    'tub-1': (0.30135634, 0.31351379, 0.08338263), 
                    'unc-25': (0.30731042, 0.252101  , 0.08433111), 
                    'unc-43': (0.290434  , 0.19049753, 0.09402576), 
                    'unc-49': (0.25054059, 0.13356404, 0.10297563), 
                    'unc-77': (0.19195101, 0.08505507, 0.10221247), 
                    'unc-80': (0.12274041, 0.04688919, 0.08500218)
                    }
    # heatmap_meta_df = meta_df.loc[meta_df['worm_gene'] != 'N2']
    
    clustered_features = make_clustermaps(featZ=featZ,
                                          meta=meta_df,
                                          featsets=featsets,
                                          strain_lut=heatmap_strain_cmap,
                                          feat_lut=feat_lut,
                                          group_vars=['worm_gene'], #'imaging_date_yyyymmdd'
                                          saveto=save_clustermap)
    plt.show()
    plt.close('all')
    #%% make clustermap as above, but with strain names insead of colourbar
    # TODO:Format text properly on clustermap!
    save_clustermap_named = saveto / 'all_strains_clustermaps_named'
    clustered_features = make_paper_clustermaps(featZ=featZ,
                                          meta=meta_df,
                                          featsets=featsets,
                                          strain_lut=heatmap_strain_cmap,
                                          feat_lut=feat_lut,
                                          group_vars=['worm_gene'], #'imaging_date_yyyymmdd'
                                          saveto=save_clustermap_named)
    plt.show()
    plt.close('all')
    
    #%% Make heatmap of strains showing number of significant features- should be able to define order (not possible using clustermap)
    # To make heatmap easy to interpret I set values to either 1 or 0
    # This means that it can be coloured as black-white for sig/non-sig feats
    combined_strain_stat_copy = combined_strains_stats
    heatmap_stats = combined_strain_stat_copy.fillna(value=1)
    heatmap_stats = heatmap_stats.apply(lambda x: [y if y >= 0.05 else 0 for y in x])
    
    hm_colors = ((0.0, 0.0, 0.0), (0.95, 0.95, 0.9))
    hm_cmap = LinearSegmentedColormap.from_list('Custom', 
                                                hm_colors, 
                                                len(hm_colors))
    
    plt.subplots(figsize=[7.5,5])
    plt.gca().yaxis.tick_right()
    plt.yticks(fontsize=9)
    
    ax=sns.heatmap(data=heatmap_stats,
                    vmin=0,
                    vmax=0.5,
                    xticklabels=False,
                    yticklabels=True,
                    cbar_kws = dict(use_gridspec=False,location="top"),
                    cmap=hm_cmap
                    )
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.1,  0.4])
    colorbar.set_ticklabels(['P < 0.05', 'P > 0.05'])
    
    ax.set_ylabel('')

    plt.savefig(saveto / 'ordered_formatted_heatmap.png', 
                bbox_inches='tight', dpi=300)

    #%% now get an idea of N2 robustness - measure Coefficient of Variance
    
    cov_by_date = []
    for g in genes:
        _meta = meta_df.query('@g in worm_gene')
        
        _feat_grouped = pd.concat([feat_nonan.loc[_meta.index],
                             _meta], axis=1).groupby('date_yyyymmdd').mean()
        
        _feat_cov = _feat_grouped[featsets['all']].apply(stats.variation).apply(np.abs)
        # pd.Series(np.abs(stats.variation(_feat_grouped[featsets['all']])),
        #                       index=featsets['all'])
        _feat_cov['worm_gene'] = g
        cov_by_date.append(_feat_cov.to_frame().transpose())

    cov_by_date = pd.concat(cov_by_date)
    cov_by_date.set_index('worm_gene',inplace=True)
    
    #find mean and std for cov for N2 by resampling
    n_per_strains = meta.query('@genes in worm_gene').groupby('worm_gene').apply(len)
    
    #10 iterations of resampling
    control_cov = []
    for i in range(0,10):
        _meta = meta_df.query('@CONTROL_STRAIN in worm_gene').sample(int(n_per_strains.mean()))
        
        _feat_grouped = pd.concat([feat_nonan.loc[list(_meta.index),:],
                             _meta], axis=1).groupby('date_yyyymmdd').mean()
        
        _feat_cov = _feat_grouped[featsets['all']].apply(stats.variation).apply(np.abs)
        # pd.Series(np.abs(stats.variation(_feat_grouped[featsets['all']])),
        #                       index=featsets['all'])
        control_cov.append(_feat_cov.to_frame().transpose())
    
    control_cov = pd.concat(control_cov)
    # genes_copy = genes
    # coV_genes = genes_copy.remove('N2')
    
    fig, ax = plt.subplots(figsize=(10,8))
    # plt.errorbar(cov_by_date.index, cov_by_date.mean(axis=1), cov_by_date.std(axis=1)/(cov_by_date.shape[1]**-2))
    cov_by_date.mean(axis=1).plot(color=(0.2, 0.2, 0.2))
    ax.set_xticks(range(0, len(genes)))  
    ax.set_xticklabels(labels=genes,
                       rotation=90)
    ax.set_ylabel('Coefficient of Variation')
    
    plt.plot(np.zeros(len(genes))+control_cov.mean(axis=1).mean(),
             '--', color=strain_lut[CONTROL_STRAIN])
    plt.fill_between([x.get_text() for x in ax.axes.xaxis.get_ticklabels()],
                     np.zeros(len(genes))+control_cov.mean(axis=1).mean() - control_cov.mean(axis=1).std(),
                     np.zeros(len(genes))+control_cov.mean(axis=1).mean() + control_cov.mean(axis=1).std(),
                     color=strain_lut[CONTROL_STRAIN],
                     alpha=0.7)
    plt.tight_layout()
    plt.savefig(saveto / 'Acoefficient_of_variation.png', dpi=300)
    plt.savefig(saveto / 'Acoefficient_of_variation.svg', dpi=300)



    #%% PCA plots by strain - prestim, bluelight and poststim separately
    # do PCA on entire space and plot worms as they travel through
 
    # Make long form feature matrix
    long_featmat = []
    for stim,fset in featsets.items():
        if stim != 'all':
            _featmat = pd.DataFrame(data=feat.loc[:,fset].values,
                                    columns=['_'.join(s.split('_')[:-1])
                                               for s in fset],
                                    index=feat.index)
            _featmat['bluelight'] = stim
            _featmat = pd.concat([_featmat,
                                  meta.loc[:,'worm_gene']],
                                 axis=1)
            long_featmat.append(_featmat)
    long_featmat = pd.concat(long_featmat,
                             axis=0)
    long_featmat.reset_index(drop=True,
                             inplace=True)
    
    full_fset = list(set(long_featmat.columns) - set(['worm_gene', 'bluelight']))
    long_feat_nonan = impute_nan_inf(long_featmat[full_fset])

    long_meta = long_featmat[['worm_gene', 'bluelight']]
    long_featmatZ = pd.DataFrame(data=stats.zscore(long_feat_nonan[full_fset], axis=0),
                                 columns=full_fset,
                                 index=long_feat_nonan.index)
    
    assert long_featmatZ.isna().sum().sum() == 0
     #%% Generate PCAs
    pca = PCA()
    X2=pca.fit_transform(long_featmatZ.loc[:,full_fset])

    # Explain PC variance using cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    thresh = cumvar <= 0.95 #set 95% variance threshold
    cut_off = int(np.argwhere(thresh)[-1])

    # #Plot as figure
    plt.figure()
    plt.plot(range(0, len(cumvar)), cumvar*100)
    plt.plot([cut_off,cut_off], [0, 100], 'k')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('variance explained')
    plt.tight_layout()
    plt.savefig(saveto / 'long_df_variance_explained.png', dpi =300)
    
    # #now put the 1:cut_off PCs into a dataframe
    PCname = ['PC_%d' %(p+1) for p in range(0,cut_off+1)]
    PC_df = pd.DataFrame(data=X2[:,:cut_off+1],
                          columns=PCname,
                          index=long_featmatZ.index)

    PC_plotting = pd.concat([PC_df,
                              long_meta[['worm_gene',
                                            'bluelight']]],
                              axis=1)
    
    # groupby worm gene to see the trajectory through PC space
    PC_plotting_grouped = PC_plotting.groupby(['worm_gene',
                                                'bluelight']).mean().reset_index()
    PC_plotting_grouped['stimuli_order'] = PC_plotting_grouped['bluelight'].map(STIMULI_ORDER)
    PC_plotting_grouped.sort_values(by=['worm_gene',
                                        'stimuli_order'],
                                    ascending=True,
                                    inplace=True)
    
    
    # Calculate standard error of mean of PC matrix computed above
    PC_plotting_sd = PC_plotting.groupby(['worm_gene',
                                          'bluelight']).sem().reset_index()
    # Map to stimuli order of PC_Grouped dataframe
    PC_plotting_sd['stimuli_order'] = PC_plotting_sd['bluelight'].map(STIMULI_ORDER)
    
#%% Make PC plots of all strains
    save_PCA = saveto / 'PCA_plots'

    plt.figure(figsize = [14,12])    
    s=sns.scatterplot(x='PC_1',
                    y='PC_2',
                    data=PC_plotting_grouped,
                    hue='worm_gene',
                    style='bluelight',
                    style_order=STIMULI_ORDER.keys(),
                    hue_order=strain_lut.keys(),
                    palette=strain_lut,
                    linewidth=0,
                    s=70)
    s.errorbar(
                x=PC_plotting_grouped['PC_1'],
                y=PC_plotting_grouped['PC_2'],
                xerr=PC_plotting_sd['PC_1'], 
                yerr=PC_plotting_sd['PC_2'],
                fmt='.',
                alpha=0.2,
                )
    ll=sns.lineplot(x='PC_1',
                y='PC_2',
                data=PC_plotting_grouped,
                hue='worm_gene',
                hue_order=strain_lut.keys(),
                palette=strain_lut,
                alpha=0.8,
                legend=False,
                sort=False)    
    plt.autoscale(enable=True, axis='both')
    # plt.axis('equal')
    plt.legend(loc='right', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5), fontsize='large')
    plt.xlabel('PC_1 ({}%)'.format(np.round(pca.explained_variance_ratio_[2]*100,2)))
    plt.ylabel('PC_2 ({}%)'.format(np.round(pca.explained_variance_ratio_[3]*100,2)))                                 
    plt.tight_layout()
    plt.savefig(save_PCA / 'PC1PC2_trajectory_space.png', dpi=400)
  
    plt.figure(figsize = [14,12])    
    s=sns.scatterplot(x='PC_3',
                    y='PC_4',
                    data=PC_plotting_grouped,
                    hue='worm_gene',
                    style='bluelight',
                    style_order=STIMULI_ORDER.keys(),
                    hue_order=strain_lut.keys(),
                    palette=strain_lut,
                    linewidth=0,
                    s=70)
    s.errorbar(
                x=PC_plotting_grouped['PC_3'],
                y=PC_plotting_grouped['PC_4'],
                xerr=PC_plotting_sd['PC_3'], 
                yerr=PC_plotting_sd['PC_4'],
                fmt='.',
                alpha=0.2,
                )
    ll=sns.lineplot(x='PC_3',
                y='PC_4',
                data=PC_plotting_grouped,
                hue='worm_gene',
                hue_order=strain_lut.keys(),
                palette=strain_lut,
                alpha=0.8,
                legend=False,
                sort=False)    
    plt.autoscale(enable=True, axis='both')
    # plt.axis('equal')
    plt.legend(loc='right', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5), fontsize='large')
    plt.xlabel('PC_3 ({}%)'.format(np.round(pca.explained_variance_ratio_[2]*100,2)))
    plt.ylabel('PC_4 ({}%)'.format(np.round(pca.explained_variance_ratio_[3]*100,2)))                                 
    plt.tight_layout()
    plt.savefig(save_PCA / 'PC3PC4_trajectory_space.png', dpi=400)
    
    #%% Make a PC plot of  strains that show a dfferent response to bluelight
    # I.e. bbs-1, bbs-2, unc-43 and possibly unc-80
    # Make change the colourmap to select the above strains only
    PC_cmap =  {
                    'N2': ('royalblue'), 
                    'add-1': ('lightgrey'), 
                    'avr-14': ('lightgrey'), 
                    'bbs-1': ('lightcoral'),
                    'bbs-2': ('maroon'),
                    # 'bbs-1': (0.94962434, 0.86199684, 0.76700144), 
                    # 'bbs-2': (0.95486878, 0.80573384, 0.75808215), 
                    'cat-2': ('lightgrey'), 
                    'cat-4': ('lightgrey'), 
                    'dys-1': ('lightgrey'), 
                    'figo-1': ('lightgrey'), 
                    'glc-2': ('lightgrey'), 
                    'glr-1': ('lightgrey'), 
                    'glr-4': ('lightgrey'), 
                    'gpb-2': ('lightgrey'), 
                    'kcc-2': ('lightgrey'), 
                    'mpz-1': ('lightgrey'), 
                    'nca-2': ('lightgrey'), 
                    'pink-1': ('lightgrey'), 
                    'snf-11': ('lightgrey'), 
                    'snn-1': ('lightgrey'), 
                    'tmem-231': ('lightgrey'), 
                    'tub-1': ('lightgrey'), 
                    'unc-25': ('lightgrey'), 
                    'unc-43': ('red'),
                    # 'unc-43': (0.290434  , 0.19049753, 0.09402576), 
                    'unc-49': ('lightgrey'), 
                    'unc-77': ('lightgrey'), 
                    'unc-80': ('chocolate'),
                    # 'unc-80': (0.12274041, 0.04688919, 0.08500218),
                    }
    
    # Plot PC1 and PC2
    plt.figure(figsize = [14,12])

    sp=sns.scatterplot(x='PC_1',
                    y='PC_2',
                    data=PC_plotting_grouped,
                    hue='worm_gene',
                    style='bluelight',
                    style_order=STIMULI_ORDER.keys(),
                    hue_order=PC_cmap.keys(),
                    palette=PC_cmap,
                    linewidth=0,
                    s=70,
                    )
    
    sp.errorbar(
                x=PC_plotting_grouped['PC_1'],
                y=PC_plotting_grouped['PC_2'],
                xerr=PC_plotting_sd['PC_1'], 
                yerr=PC_plotting_sd['PC_2'],
                fmt='.',
                alpha=0.2,
                )    
    
    l=sns.lineplot(x='PC_1',
                    y='PC_2',
                    data=PC_plotting_grouped,
                    hue='worm_gene',
                    hue_order=PC_cmap.keys(),
                    palette=PC_cmap,
                    alpha=0.8,
                    legend=False,
                    sort=False,)    
        
    plt.autoscale(enable=True, axis='both')
    plt.legend(loc='right', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5), fontsize='large')
    plt.xlabel('PC_1 ({}%)'.format(np.round(pca.explained_variance_ratio_[0]*100,2)))
    plt.ylabel('PC_2 ({}%)'.format(np.round(pca.explained_variance_ratio_[1]*100,2)))                                 
    plt.tight_layout()

    plt.savefig(save_PCA / 'different_bluelight_response_PC1PC2_trajectory_space.png', dpi=400)
    
    # Plot PC3 and PC4
    plt.figure(figsize = [14,12])

    sp=sns.scatterplot(x='PC_3',
                    y='PC_4',
                    data=PC_plotting_grouped,
                    hue='worm_gene',
                    style='bluelight',
                    style_order=STIMULI_ORDER.keys(),
                    hue_order=PC_cmap.keys(),
                    palette=PC_cmap,
                    linewidth=0,
                    s=70,
                    )
    
    sp.errorbar(
                x=PC_plotting_grouped['PC_3'],
                y=PC_plotting_grouped['PC_4'],
                xerr=PC_plotting_sd['PC_3'], 
                yerr=PC_plotting_sd['PC_4'],
                fmt='.',
                alpha=0.2,
                )    
    
    l=sns.lineplot(x='PC_3',
                    y='PC_4',
                    data=PC_plotting_grouped,
                    hue='worm_gene',
                    hue_order=PC_cmap.keys(),
                    palette=PC_cmap,
                    alpha=0.8,
                    legend=False,
                    sort=False,)    
        
    plt.autoscale(enable=True, axis='both')
    plt.legend(loc='right', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5), fontsize='large')
    plt.xlabel('PC_3 ({}%)'.format(np.round(pca.explained_variance_ratio_[0]*100,2)))
    plt.ylabel('PC_4 ({}%)'.format(np.round(pca.explained_variance_ratio_[1]*100,2)))                                 
    plt.tight_layout()

    plt.savefig(save_PCA / 'different_bluelight_response_PC3PC4_trajectory_space.png', dpi=400)

#%%
    save_box = saveto / 'example_feature_boxplots'
    
    for k,v in EXAMPLES.items():
        examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
                                                                            CONTROL_STRAIN,
                                                                            feat_df=feat,
                                                                            meta_df=meta)
    
        # filter features
        examples_feat_df, examples_meta_df, featsets = filter_features(examples_feat_df,
                                                     examples_meta_df)


        examples_strain_lut = make_colormaps(gene_list,
                                                featlist=featsets['all'],
                                                idx=dx,
                                                candidate_gene=v
                                                )
        examples_strain_lut = examples_strain_lut[0]
    
        feature_box_plots(k,
                          examples_feat_df,
                          examples_meta_df,
                          examples_strain_lut,
                          show_raw_data=True,
                          add_stats=True)
        plt.tight_layout()
        plt.savefig(save_box / '{}_boxplot.png'.format(k), 
                    bbox_inches="tight",
                    dpi=400)
        plt.close('all')

  #%% TODO: Write nice script that imports the precomputed pValues into the boxplots
  # This will be super important for the paper figures (right now I've devised a 
  # hacky workaround by changing the custom annot section of statannot but this utilises
  # KW to perform stats and make the plots...) Need to think of an iterator solution
  # that can fit with the 'box_pairs' input of the functions below!!
   
    # save_box = saveto / 'example_feature_boxplots'
  
    # for k,v in EXAMPLES.items():
    #     examples_feat_df, examples_meta_df, dx, gene_list = select_strains(v,
    #                                                                         CONTROL_STRAIN,
    #                                                                         feat_df=feat_df,
    #                                                                         meta_df=meta_df)
    
    #     # filter features
    #     examples_feat_df, examples_meta_df, featsets = filter_features(examples_feat_df,
    #                                                   examples_meta_df)

    #     examples_strain_lut = make_colormaps(gene_list,
    #                                             featlist=featsets['all'],
    #                                             idx=dx,
    #                                             candidate_gene=v
    #                                             )
    #     examples_strain_lut = examples_strain_lut[0]
    #     example_strains = v
    #     example_bhP_values = combined_strains_stats.loc[(example_strains)]
    #     selected_feature_bhP_values = [example_bhP_values]
        
    #     # [combined_strains_stats[index].isin([example_strains])]    

        
    #    # example_bhp_values =  combined_strains_stats
    
    #    #  example_bhP_values = pd.read_csv(saveto / '{}_stats.csv'.format(candidate_gene),
    #    #                                       index_col=False)
    #    #  example_bhP_values.rename(mapper={0:'p<0.05'},
    #    #                                inplace=True)    
    
    #     feature_box_plots(feature=k,
    #                       feat_df=examples_feat_df,
    #                       meta_df=examples_meta_df,
    #                       strain_lut=examples_strain_lut,
    #                       show_raw_data=True,
    #                       add_stats=True,
    #                       # bhP_values_df=None,
    #                       # statistics=selected_feature_bhP_values
    #                       )
    #     plt.tight_layout()
    #     plt.savefig(save_box / '{}_boxplot.png'.format(k), 
    #                 bbox_inches="tight",
    #                 dpi=400)
    #     plt.close('all')

