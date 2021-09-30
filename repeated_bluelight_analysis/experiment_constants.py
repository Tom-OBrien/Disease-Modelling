#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:54:48 2021

@author: lferiani
"""

import numpy as np

STIM_START_S = {'10x': 300+np.arange(10)*100, '20x': 300+np.arange(20)*100}
STIM_DURATION_S = 10
FPS = 25

MD_COLS = [
    'date_yyyymmdd',
    'imaging_run_number',
    'imaging_plate_id',
    'imgstore_name',
    'camera_serial',
    'worm_strain',
    'led_intensity',
    'well_name',
    ]

HIFPS_COLS = [
    'worm_index',
    'well_name',
    'timestamp',
    'speed',
    'd_speed',
    'length',
    'width_head_base',
    'width_midbody',
    'd_speed_midbody',
    'motion_mode',
    ]

IMAGINGRUN2TIMEPOINT = {
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    }

