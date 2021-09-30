#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:25:36 2021

@author: tobrien

Script for compiling metadata and adding well annotions to denote bad wells
for disease modelling work

"""

import pandas as pd
from pathlib import Path
import re

from tierpsytools.hydra.compile_metadata import populate_96WPs,\
    get_day_metadata, concatenate_days_metadata, day_metadata_check, \
        number_wells_per_plate

date_regex = r"\d{8}"

PROJECT_DIRECTORY = Path('/Volumes/behavgenom$/Tom/Data/Hydra/RepeatedBluelight/RawData')


#%%
if __name__ == '__main__':

    day_root_dirs = [d for d in (PROJECT_DIRECTORY /
                                 'AuxiliaryFiles').glob("*")
                     if d.is_dir() and re.search(date_regex, str(d))
                     is not None]

    print('Calculating metadata for {} days of experiments'.format(
            len(day_root_dirs)))

    for count, day in enumerate(day_root_dirs):
        exp_date = re.findall(date_regex, str(day))[0]
        manualmetadata_file = list(day.rglob('*_manual_metadata.csv'))[0]
        assert (exp_date in str(manualmetadata_file))
        wormsorter_file = list(day.rglob('*_wormsorter.csv'))[0]
        assert (exp_date in str(wormsorter_file))

        print('Collating manual metadata files: {}'.format(wormsorter_file))

        plate_metadata = populate_96WPs(wormsorter_file,
                                        del_if_exists=True,
                                        saveto='default')


        metadata_file = day / '{}_day_metadata.csv'.format(exp_date)

        print('Generating day metadata: {}'.format(
                metadata_file))

        day_metadata = get_day_metadata(plate_metadata,
                                        manualmetadata_file,
                                        saveto=metadata_file,
                                        del_if_exists=True)

   
        files_to_check = day_metadata_check(day_metadata, day, plate_size=48)
        number_wells_per_plate(day_metadata, day)

# %%
    import datetime
    # combine all the metadata files
    concat_meta = concatenate_days_metadata(PROJECT_DIRECTORY / 'AuxiliaryFiles',
                                            list_days=None,
                                            saveto=None)
    
    
    concat_meta_grouped = concat_meta.groupby('worm_gene')

    strains = pd.DataFrame(concat_meta_grouped.apply(lambda x: x.drop_duplicates(subset='worm_strain')))
    strains.reset_index(drop=True,
                        inplace=True)
    
    strains = strains[['worm_gene',
                        'worm_strain',
                        'worm_code',
                        'date_yyyymmdd']]
 
    strains.to_csv(PROJECT_DIRECTORY / \
                   '{}_strain_name_errors.csv'.format(
                       datetime.datetime.today().strftime('%Y%m%d')
                       ),
                   index=False)

#%%

from tierpsytools.hydra.match_wells_annotations import import_wells_annotations_in_folder, update_metadata # if your tierpsytools is super up to date, the typo in the first function has been fixed 
aux_dir = Path('/Volumes/behavgenom$/Tom/Data/Hydra/RepeatedBluelight/RawData/AuxiliaryFiles')
annotations_df = import_wells_annotations_in_folder(aux_dir)
annotations_df.rename(columns={'imgstore_prestim':'imgstore'}, inplace=True)
annotations_df['is_bad_well'] = annotations_df['well_label'] != 1
wells_annotated_metadata = update_metadata(aux_dir, annotations_df, del_if_exists=True)        