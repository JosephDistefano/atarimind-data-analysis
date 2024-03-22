import json
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import csv
import time
import datetime

def extract_all_data_per_frame(config):
    for game in config['games']:
        print(game)
        for subject in config['subjects']:
            read_path = config['processed_data_path'] + game + '/game/' 
            files = os.listdir(read_path)
            for file in files:
                sub = file[0:2]
                session = file[3:10]
                process_all_data_per_frame_path = config['processed_data_path'] + game + '/combined/' + sub + '_' + session + '_all_features_per_frame.csv'
                if sub == 'JD': 
                    b_alert_subject_name = config['b_alert_subjects'][0]
                    with open(process_all_data_per_frame_path, 'w') as file: 
                        dw = csv.DictWriter(file, delimiter=',',  
                                            fieldnames=config['header_list_all_data_per_frame']) 
                        dw.writeheader() 
  
                eye_path = config['processed_data_path'] + game + '/eye/' + sub + '_' + session + '_eye_features.csv'
                eeg_path = config['processed_data_path'] + game + '/eeg/' +  b_alert_subject_name + '00000_' +session + '.Classification.csv'
                game_path = config['processed_data_path'] + game + '/game/' + sub + '_' + session + '_game_features.csv'
                CCL_path =  config['processed_data_path'] + game + '/CCL/' + sub + '_' + session + '_CCL_features_per_frame.csv'
                eye_data = pd.read_csv(eye_path)
                eeg_data = pd.read_csv(eeg_path)
                game_data = pd.read_csv(game_path)
                CCL_data = pd.read_csv(CCL_path)
                time_stamps_game,action,shift = process_game_data_per_frame(game_data)
                time_stamps_eye,x_pos,y_pos = process_eye_data_per_frame(eye_data)
                time_stamps_eeg,ProbDistraction,ProbLowEng,ProbHighEng,ProbAveWorkload = process_eeg_data_per_frame(eeg_data)
                time_stamps_ccl,objects_x_start,objects_y_start,objects_width,objects_height,objects_centroid_x,objects_centroid_y,objects_area = process_ccl_per_frame(CCL_data)
                game_time_stamps_in_seconds = [x - time_stamps_game[0] for x in time_stamps_game] 
                eeg_time_stamps_in_seconds = []
                for t in time_stamps_eeg:
                    x = time.strptime(t[:-4].split(',')[0],'%H:%M:%S')
                    eeg_time_stamps_in_seconds.append(datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())
                et = 0
                for i in range(0,len(time_stamps_game)-1):
                    start_game = time_stamps_game[i]        
                    end_game = time_stamps_game[i+1]
                    start_eye = time_stamps_eye.index(min(time_stamps_eye, key=lambda x:abs(x-start_game)))       
                    end_eye = time_stamps_eye.index(min(time_stamps_eye, key=lambda x:abs(x-end_game)))  
                    x_pos_f = x_pos[start_eye:end_eye] 
                    y_pos_f = y_pos[start_eye:end_eye]
                    time_stamps_eye_f =time_stamps_eye[start_eye:end_eye]
                    action_f = action[i]
                    shift_f = shift[i]
                    objects_x_start_f = objects_x_start[i]

                
                    objects_y_start_f = objects_y_start[i]
                    objects_width_f = objects_width[i]
                    objects_height_f = objects_height[i]
                    objects_centroid_x_f = objects_centroid_x[i]
                    objects_centroid_y_f = objects_centroid_y[i]
                    objects_area_f =objects_area[i]
                    
                    time_stamp_in_seconds = time_stamps_game[i] -time_stamps_game[0]

                    if game_time_stamps_in_seconds[i] < eeg_time_stamps_in_seconds[et]:
                        ProbDistraction_f = ProbDistraction[et]
                        ProbLowEng_f = ProbLowEng[et]
                        ProbHighEng_f =ProbHighEng[et]
                        ProbAveWorkload_f = ProbAveWorkload[et]

                    if game_time_stamps_in_seconds[i] > eeg_time_stamps_in_seconds[et]:
                        et = et+1
                    row = [time_stamps_game[i],time_stamp_in_seconds, action_f,shift_f, time_stamps_eye_f,x_pos_f,y_pos_f,ProbDistraction_f,ProbLowEng_f,ProbHighEng_f,ProbAveWorkload_f,objects_x_start_f,objects_y_start_f,objects_width_f,objects_height_f ,objects_centroid_x_f,objects_centroid_y_f ,objects_area_f]

                    with open(process_all_data_per_frame_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row)
    return
    
    
def process_game_data_per_frame(game_data):
    time_stamps = game_data.iloc[:,0].to_list()
    action = game_data.iloc[:,1].to_list()
    shift = game_data.iloc[:,2].to_list()
    return time_stamps,action,shift

def process_eye_data_per_frame(eye_data):
    time_stamps = eye_data.iloc[:,0].to_list()
    x_pos = eye_data.iloc[:,1].to_list()
    y_pos = eye_data.iloc[:,2].to_list()
    return time_stamps,x_pos,y_pos

def process_ccl_per_frame(ccl_data):
    time_stamps = ccl_data.iloc[:,0].to_list()
    objects_x_start=ccl_data.iloc[:,1].to_list()
    objects_y_start=ccl_data.iloc[:,2].to_list()
    objects_width=ccl_data.iloc[:,3].to_list()
    objects_height=ccl_data.iloc[:,4].to_list()
    objects_centroid_x=ccl_data.iloc[:,5].to_list()
    objects_centroid_y=ccl_data.iloc[:,6].to_list()
    objects_area = ccl_data.iloc[:,7].to_list()
    return time_stamps,objects_x_start,objects_y_start,objects_width,objects_height,objects_centroid_x,objects_centroid_y,objects_area

def process_eeg_data_per_frame(eeg_data):
    time_stamps = eeg_data["Elapsed Time"].to_list()
    ProbDistraction = eeg_data["ProbDistraction"].to_list()
    ProbLowEng = eeg_data["ProbLowEng"].to_list()
    ProbHighEng = eeg_data["ProbHighEng"].to_list()
    # CogState = eeg_data["CogState"].to_list()
    # ProbFBDSWorkload = eeg_data["ProbFBDSWorkload"].to_list()
    # ProbBDSWorkload = eeg_data["ProbBDSWorkload"].to_list()
    ProbAveWorkload = eeg_data["ProbAveWorkload"].to_list()
    return time_stamps,ProbDistraction,ProbLowEng,ProbHighEng,ProbAveWorkload
