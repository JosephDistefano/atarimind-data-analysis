import numpy as np
import pandas as pd
import os
from pathlib import Path
import math
import json
from .mne_import_xdf import read_raw_xdf
from .utils import nested_dict
import ujson
from PIL import Image
import csv
import pyxdf
import matplotlib

def read_xdf_game_data(config, file,game):
    xdf_file  = file 
    read_path = config['raw_data_path'] + game + '/' + xdf_file
    streams, fileheader = pyxdf.load_xdf(read_path)
    for stream in streams:
        if stream["info"]["name"][0] == 'game_states':
            raw_game =  stream["time_series"]
            game_time_stamps =  stream["time_stamps"]
    return raw_game, game_time_stamps

def extract_game_data_and_frames(config):
    for game in config['games']:
        read_path = config['raw_data_path'] + game + '/' 
        files = os.listdir(read_path)
        for file in files:
            sub = file[4:6]
            session = file[-15:-8]
            game_data, game_time_info = read_xdf_game_data(config,file,game)
            extract_game_features(config,game,sub,session,game_data,game_time_info)
    return 

def extract_game_features(config,game,sub,session,game_data,game_time_info):
    game_feature_path = config['extracted_game_features_data_path'] + game + '/game/' +sub + '_'+session+ '_game_features.csv'
    frame_path = config['extracted_frames_data_path'] + game + '/frame/' +sub + '_'+ session +'_'
    for i in range(0,len(game_data)):
        frame_path_img = frame_path + str(game_time_info[i]) + '.png'
        d = ujson.loads(game_data[i][0])
        frame = d['frame']
        action = d['action']
        shift = d['shift']
        frame = np.asarray(frame)
        im = Image.fromarray((frame * 255).astype(np.uint8))
        im.save(frame_path_img)
        row = [game_time_info[i],action,shift]
        with open(game_feature_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    return


# def extract_x_y_eye_location_and_save(config,data):
#     eye_x_y_time_dict = {}
#     game_frames_dict = {}
#     game_action_dict = {}
#     for game in config['games']:
#         print(game)
#         eye_x_y_time_dict[game] = {}
#         game_frames_dict[game] = {}
#         game_action_dict[game] = {}
#         all_data= data[game]
#         for key in all_data.keys():
#             eye_data = all_data[key]['eye']['time_series']
#             eye_data_time_stamps = all_data[key]['eye']['time_stamps']
#             eye_x_y_time= process_eye(config,eye_data,eye_data_time_stamps)
#         eye_x_y_time_dict[game][key] = eye_x_y_time
#     return eye_x_y_time_dict # ,game_action_dict,game_frames_dict 

