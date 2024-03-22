import numpy as np
import pandas as pd
import os
from pathlib import Path
import math
import json
from .mne_import_xdf import read_raw_xdf
from .utils import nested_dict
import pyxdf
import csv

def read_xdf_eye_data(config, file,game):
    xdf_file  = file 
    read_path = config['raw_data_path'] + game + '/' + xdf_file
    streams, fileheader = pyxdf.load_xdf(read_path)
    raw_game, time_info = None, None
    for stream in streams:
        if stream["info"]["name"][0] == 'Tobii_Eye_Tracker':
            raw_eye = stream["time_series"]
            eye_time_stamps =  stream["time_stamps"]
    return raw_eye, eye_time_stamps

def extract_eye_data_and_features(config):
    all_game_data = {}
    for game in config['games']:
        read_path = config['raw_data_path'] + game + '/' 
        files = os.listdir(read_path)
        for file in files:
            sub = file[4:6]
            session = file[-15:-8]
            eye_data, eye_time_info = read_xdf_eye_data(config,file, game)
            extract_eye_features(config,game,sub,session,eye_data,eye_time_info)
    return 

def extract_eye_features(config,game,sub,session,eye_data,eye_time_info):
    eye_feature_path = config['extracted_eye_features_data_path'] + game + '/eye/' +sub + '_'+session+ '_eye_features.csv'
    screen_width = config['screen_size'][0]
    screen_height = config['screen_size'][1]
    atari_width = config['atari_size'][0]
    atari_height = config['atari_size'][1]
    top_left_corner_x = screen_width/2 - atari_width/2
    top_left_corner_y = screen_height/2 - atari_height/2
    top_right_corner_x = top_left_corner_x +atari_width
    bottom_right_corner_y = top_left_corner_y + atari_height
    for i in range(0,len(eye_time_info)):
        d = json.loads(eye_data[i][0])
        x = d['AvgGazePointX']
        y = d['AvgGazePointY']
        if math.isnan(x):
            x = 0 
        if math.isnan(y):
            y = 0
        x = int(x * screen_width)
        y = int(y * screen_height) 
        if x < top_left_corner_x:
            x = 0
            y = 0
        if top_right_corner_x < x:
            x = 0
            y = 0
        if y < top_left_corner_y:
            y = 0
            x = 0
        if bottom_right_corner_y < y:
            y = 0
            x = 0
        if x != 0:
            x = x - top_left_corner_x
            x = int(x/3)
        if y!= 0:
            y = y- top_left_corner_y   
            y = int(y/3)
        row = [eye_time_info[i],x,y]
        with open(eye_feature_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    return 

