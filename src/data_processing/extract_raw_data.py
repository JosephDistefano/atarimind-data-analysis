import pyxdf
import numpy as np
import pandas as pd
import ujson
import os
from pathlib import Path
import math
import json
from .mne_import_xdf import read_raw_xdf
import ijson
from numba import njit, cuda
from .utils import nested_dict

def read_xdf_eeg_data(config, file, game):
    xdf_file  = file 
    read_path = config['raw_data_path'] + game + '/' + xdf_file
    raw_eeg, eeg_time_stamps = read_raw_xdf(read_path)
    return raw_eeg, eeg_time_stamps

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

def read_xdf_game_data(config, file,game):
    xdf_file  = file 
    read_path = config['raw_data_path'] + game + '/' + xdf_file
    streams, fileheader = pyxdf.load_xdf(read_path)
    for stream in streams:
        if stream["info"]["name"][0] == 'game_states':
            raw_game = [ujson.loads(data[0]) for data in stream["time_series"]]
            game_time_stamps =  stream["time_stamps"]
    return raw_game, game_time_stamps

def extract_all_data_from_xdf_save_to_h5(config):
    all_game_data = {}
    for game in config['games']:
        data = nested_dict()
        read_path = config['raw_data_path'] + game + '/' 
        files = os.listdir(read_path)
        x = 0
        play_session = 'play_session' + str(x)
        for file in files:
            eye_data, eye_time_info = read_xdf_eye_data(config,
                file, game)
                
            # eeg_data, eeg_time_info = read_xdf_eeg_data(
            #     config,file,game)
            game_data, game_time_info = read_xdf_game_data(
                 config,file,game)

            data[play_session]['eye']['time_series'] = eye_data
            # data[play_session]['eeg']['time_series'] = eeg_data
            data[play_session]['game']['time_series'] = game_data

            data[play_session]['eye']['time_stamps'] = eye_time_info
            # data[play_session]['eeg']['time_stamps'] = eeg_time_info
            data[play_session]['game']['time_stamps'] = game_time_info
            print(game)
            x = x+1
        all_game_data[game] = data

    return all_game_data

def extract_x_y_eye_location_and_save(config,data):
    eye_x_y_time_dict = {}
    game_frames_dict = {}
    game_action_dict = {}
    for game in config['games']:
        eye_x_y_time_dict[game] = {}
        game_frames_dict[game] = {}
        game_action_dict[game] = {}
        all_data= data[game]
        for key in all_data.keys():
            eye_data = all_data[key]['eye']['time_series']
            eye_data_time_stamps = all_data[key]['eye']['time_stamps']
            # eeg_data = all_data[key]['eeg']['time_series']
            # eeg_data_time_stamps = all_data[key]['eeg']['time_stamps']
            # game_data = all_data[key]['game']['time_series']
            # game_time_stamps = all_data[key]['game']['time_stamps']
            eye_x_y_time= process_eye(config,eye_data,eye_data_time_stamps)
            # print('here')
            # game_frames_time,game_action_times = process_game(config,game_data,game_time_stamps)
            # print('here2')
        eye_x_y_time_dict[game][key] = eye_x_y_time
        # game_frames_dict[game][key] = game_frames_time
        # game_action_dict[game][key] = game_action_times
    return eye_x_y_time_dict # ,game_action_dict,game_frames_dict 

def process_eye(config,eye_data,eye_data_time_stamps):
    eye_x_y_time = {}

    screen_width = config['screen_size'][0]
    screen_height = config['screen_size'][1]
    atari_width = config['atari_size'][0]
    atari_height = config['atari_size'][1]
    top_left_corner_x = screen_width/2 - atari_width/2
    top_left_corner_y = screen_height/2 - atari_height/2
    top_right_corner_x = top_left_corner_x +atari_width
    bottom_right_corner_y = top_left_corner_y + atari_height
    for i in range(0,len(eye_data)):
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
        eye_x_y_time[eye_data_time_stamps[i]] = [x,y]
    return eye_x_y_time

def process_game(config,game_data,game_time_stamps):
    game_frames = {}
    game_actions = {}
    for i in range(0,len(game_data)):
        d= []
        game_actions[game_time_stamps[i]] = {}
        game_actions[game_time_stamps[i]] = {}
        d = ujson.loads(game_data[i][0])
        frame = d['frame']
        action = d['action']
        shift = d['shift']
        frame = np.asarray(frame)
        game_frames[game_time_stamps[i]] = frame
        game_actions[game_time_stamps[i]]['shift'] = shift
        game_actions[game_time_stamps[i]]['action'] = action
    return game_frames,game_actions
