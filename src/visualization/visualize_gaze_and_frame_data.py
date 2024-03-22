import matplotlib.pyplot as plt
import os
import pandas as pd
import json

def visualize_all_data(config):
    for game in config['games']:
        game= 'breakout'
        for subject in config['subjects']:
            read_path = config['processed_data_path'] + game + '/game/' 
            files = os.listdir(read_path)
            for file in files:
                sub = file[0:2]
                session = file[3:10]
                all_data_path = config['processed_data_path'] + game + '/combined/' + sub + '_' + session + '_all_features_per_frame.csv'
                frame_path = config['processed_data_path'] + game + '/frame/' + sub + '_' + session +'_'
                ccl_frame_path = config['processed_data_path'] + game + '/CCL_frames/' + sub + '_' + session +'_'
                all_data = pd.read_csv(all_data_path)
                time_stamps = all_data['time_stamps_game'].to_list()
                action = all_data['action'].to_list()
                shift = all_data['shift'].to_list()
                x_pos = all_data['eye_x_pos'].to_list()
                y_pos = all_data['eye_y_pos'].to_list()
                ProbDistraction = all_data['ProbDistraction'].to_list()
                ProbLowEng = all_data['ProbLowEng'].to_list()
                ProbHighEng = all_data['ProbHighEng'].to_list()
                ProbAveWorkload = all_data['ProbAveWorkload'].to_list()
                fig,axs = plt.subplots(1,5)
                for i in range(0,len(time_stamps)):
                    frame_path_f = frame_path + str(time_stamps[i]) + '.png'
                    ccl_frame_path_f = ccl_frame_path + str(time_stamps[i]) + '.png'
                    x_pos_f = [float(x) for x in x_pos[i].strip('[]').split(', ')]
                    y_pos_f = [float(x) for x in y_pos[i].strip('[]').split(', ')]
                    ProbDistraction_f = ProbDistraction[i] 
                    ProbAveWorkload_f = ProbAveWorkload[i]
                    ProbHighEng_f = ProbHighEng[i]
                    img = plt.imread(frame_path_f)
                    cimg = plt.imread(ccl_frame_path_f)
                    axs[0].imshow(img)
                    axs[0].scatter(x_pos_f,y_pos_f,c='w', s=40)
                    act_shift = 'action: ' + str(action[i]) + '  shift: ' + str(shift[i])
                    dl = 'Distraction:  ' +str(ProbDistraction_f)
                    wl = 'Workload:  ' + str(ProbAveWorkload_f)
                    hel = 'High Engagament:  ' +str(ProbHighEng_f)
                    axs[0].set_xlabel(act_shift)
                    axs[1].bar(0,ProbDistraction_f)
                    axs[1].set_xlabel(dl)
                    axs[1].set(ylim=(0,1))
                    axs[2].bar(0,ProbAveWorkload_f)
                    axs[2].set_xlabel(wl)
                    axs[2].set(ylim=(0,1))
                    axs[3].bar(0,ProbHighEng_f)
                    axs[3].set_xlabel(hel)                    
                    axs[3].set(ylim=(0,1))
                    axs[4].imshow(cimg)
                    axs[4].set_xlabel('CCL')                    
                    plt.pause(.01)

                    axs[0].cla()
                    axs[1].cla()
                    axs[2].cla()
                    axs[3].cla()
                    axs[4].cla()
    return 
