import os
import csv
from PIL import Image
import numpy as np
import os
from pathlib import Path
import pandas
import re
import cv2
import numpy as np
import ujson
import csv
import pickle   
import tarfile
import time

def extract_CCL_data(config):
    for game in config['games']:
        print(game)
        for subject in config['subjects']:
            read_path = config['processed_data_path'] + game + '/frame/' 
            files = os.listdir(read_path)
            ccl_feature_file_save_path = config['processed_data_path'] + game + '/CCL/' + files[0][0:2] + '_' +  files[0][3:10] + '_CCL_features_per_frame.csv'

            for file in files:
                sub = file[0:2]
                session = file[3:10]
                time_stamp = file[-21:-4]
                if time_stamp[0] == '_':
                    time_stamp = time_stamp[1:]
                if time_stamp[1] == '_':
                    time_stamp = time_stamp[2:]
                ccl_image_file_save_path = config['processed_data_path'] + game + '/CCL_frames/' + files[0][0:2] + '_' +  files[0][3:10] +'_' +  time_stamp +'.png'

                objects_x_start,objects_y_start,objects_width,objects_height,objects_centroid_x,objects_centroid_y,objects_area,ccl_img = process_frame_for_CCL_stats(game,config,read_path,file)

                row = [time_stamp,objects_x_start,objects_y_start,objects_width,objects_height,objects_centroid_x,objects_centroid_y,objects_area]
                with open(ccl_feature_file_save_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
                
                ccl_frame = np.asarray(ccl_img)
                ccl_frame = Image.fromarray((ccl_frame).astype(np.uint8))
                ccl_frame.save(ccl_image_file_save_path)

    return


def process_frame_for_CCL_stats(game,config,read_path,frame):
    path = read_path + frame
    print(path)
    sdfsdf


    frame = cv2.imread(path)

    ccl_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
    y_crop_start = config[game+ '_crop_x']
    x_crop_start=  config[game+ '_crop_y']
    cropped_image=ccl_img[y_crop_start:,x_crop_start:-x_crop_start]

    score_area = config[game+'_score_area']
    if game in ['breakout']:
        static_object = config[game+'_static_object']

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cropped_image , 8 , cv2.CV_32S)
    #cv2.imshow('g',np.array(labels,dtype='uint8')*255) 
    objects_x_start = []
    objects_y_start = []
    objects_width = []
    objects_height = []
    objects_centroid_x = []
    objects_centroid_y = []
    objects_area = []
    for ind in range(len(stats)):
        stat=stats[ind]
        if stat[4]<300: # Only consider game objects less than 300 pixels within the cropped area
            x_start, y_start = stat[0]+x_crop_start, stat[1]+y_crop_start
            width,height=stat[2],stat[3]
            centroid=centroids[ind]+[x_crop_start, y_crop_start]
            objects_x_start.append(x_start)
            objects_y_start.append(y_start)
            objects_width.append(width)
            objects_height.append(height)
            objects_centroid_x.append(centroid[0])
            objects_centroid_y.append(centroid[1])
            objects_area.append(stat[-1])
            ccl_img=cv2.rectangle(ccl_img, (x_start, y_start), (x_start+width, y_start+height), (255))
        
    return objects_x_start,objects_y_start,objects_width,objects_height,objects_centroid_x,objects_centroid_y,objects_area,ccl_img
