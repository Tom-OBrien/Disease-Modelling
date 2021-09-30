#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:02:53 2021

Merge my+Ida's feature summaries, filenames and metadata then filter in order
generate cluster maps and PCA plotsfor paper figures

@author: tobrien
"""
import pandas as pd
from pathlib import Path
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata

my_FEAT_FILE =  Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/features_summary_tierpsy_plate_filtered_traj_compiled.csv') 
my_FNAME_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')
my_METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/exploratory_metadata/EXPLORATORY_metadata_with_wells_annotations.csv')

ida_FEAT_FILE =  Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/summary_results_files/eleni_filters/features_summary_tierpsy_plate_filtered_traj_compiled.csv') #list(ROOT_DIR.rglob('*filtered/features_summary_tierpsy_plate_20200930_125752.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/eleni_filtered/features_summary_tierpsy_plate_filtered_traj_compiled.csv')# #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/features_summary_tierpsy_plate_20200930_125752.csv')
ida_FNAME_FILE = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/summary_results_files/eleni_filters/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')#list(ROOT_DIR.rglob('*filtered/filenames_summary_tierpsy_plate_20200930_125752.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/eleni_filtered/filenames_summary_tierpsy_plate_filtered_traj_compiled.csv')#  #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/summary_results_files/filtered/filenames_summary_tierpsy_plate_20200930_125752.csv')
ida_METADATA_FILE = Path('/Users/tobrien/OneDrive - Imperial College London/DiseaseModel_nodrugs/Data/AuxiliaryFiles/wells_updated_metadata.csv') #list(ROOT_DIR.rglob('*wells_annotated_metadata.csv'))[0] #Path('/Users/ibarlow/OneDrive - Imperial College London/Documents/behavgenom_copy/DiseaseScreen/AuxiliaryFiles/wells_annotated_metadata.csv')

SAVE_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/Disease Model Screen/Results/Combined_metadata_and_FeatureMatrix')
#%% Import my data and filter accordingly:
if __name__ == '__main__':
 
    # Read in data
    my_featMat, my_metadata = read_hydra_metadata(my_FEAT_FILE,
                                            my_FNAME_FILE,
                                            my_METADATA_FILE)
    
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
        'glc-2'
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

    # Set 'date' columns same as Ida's metadata file to merge
    my_metadata.rename(
        columns={
            'date_yyyymmdd': 'imaging_date_yyyymmdd'},
        inplace=True)

#%% Import Ida's data and filter accordingly:
    
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
        'date_yyyymmdd',
        'well_label',
        'imaging_run']

    ida_metadata = ida_metadata.drop(ida_columns_to_drop,
                                     axis=1)     
    
#%% Concatenate dataframes

    combined_metadata = pd.concat([my_metadata, ida_metadata])
    combined_featMat = pd.concat([my_featMat, ida_featMat])
    
    combined_metadata = pd.DataFrame(combined_metadata)
    combined_featMat = pd.DataFrame(combined_featMat)
    
    combined_metadata.to_csv(
        SAVE_DIR / 'Combined_DiseaseScreen_metadata.csv', index=False)
    combined_featMat.to_csv(
        SAVE_DIR / 'Combined_DiseaseScreen_FeatureMatrix.csv', index=False)
    
    