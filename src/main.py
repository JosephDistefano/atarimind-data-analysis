from pathlib import Path
import yaml
from utils import skip_run
from data_processing.extract_eye_data import extract_eye_data_and_features
from data_processing.extract_game_data import extract_game_data_and_frames
from data_processing.extract_eeg_data import write_mne_to_b_alert_edf
from data_processing.extract_data_per_frame import extract_all_data_per_frame
from data_processing.extract_data_per_epoch_length import extract_all_data_per_epoch_length
from data_processing.extract_CCL_data import extract_CCL_data
from visualization.visualize_gaze_and_frame_data import visualize_all_data


config_path = Path(__file__).parents[1] / "src/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("skip", "Extract eye data from xdf file and save features in csv") as check, check():
    extract_eye_data_and_features(config)

with skip_run("skip", "Extract game data from xdf file and save frames") as check, check():
    extract_game_data_and_frames(config)   

with skip_run("skip", "Convert EEG data to edfs for b-alert analysis") as check, check():
    write_mne_to_b_alert_edf(config,save_data=True)

with skip_run("skip", "Extract all CCL components from frame data") as check, check():
    extract_CCL_data(config)

with skip_run("run", "Epoch game data, eye data, and eeg data per frame") as check, check():
    extract_all_data_per_frame(config)   
    
with skip_run("skip", "visualize data") as check, check():
    visualize_all_data(config)




## number of windows 
## distance between top two


# distance of predicted gaze to top active window
# threshold workload and train gating entwork on this based on score
# regression with workload between 0-1. Need training from workload. utilize exact value of workload to train regression 
