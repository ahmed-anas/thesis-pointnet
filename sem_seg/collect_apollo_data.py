import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from scipy import misc

APOLLO_DATA_DEPTH_DIR = os.path.join(ROOT_DIR, 'data/annotation-apollo_scape_label-train_depth-apollo-1.5/depth')
APOLLO_DATA_LABEL_DIR = os.path.join(ROOT_DIR, 'data/annotation-apollo_scape_label-train_depth-apollo-1.5/label')

video_recording_dirs = [dir_name for dir_name in os.listdir(APOLLO_DATA_DEPTH_DIR) if os.path.isdir(os.path.join(APOLLO_DATA_DEPTH_DIR,dir_name))]



for video_recording_dir in video_recording_dirs:
    out_filename1 = video_recording_dir + '_camera5.npy'


    camera5_depth_folder_path = os.path.join(APOLLO_DATA_DEPTH_DIR, video_recording_dir, 'Camera 5')
    camera5_label_folder_path = os.path.join(APOLLO_DATA_LABEL_DIR, video_recording_dir, 'Camera 5')
    for image_name in os.listdir(camera5_depth_folder_path):

        depth_img = misc.imread(os.path.join(camera5_depth_folder_path,image_name))
        label_img = misc.imread(os.path.join(camera5_label_folder_path,image_name))

        
        


    
