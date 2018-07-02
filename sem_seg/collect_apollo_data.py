import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from scipy import misc
#import Image
import math
import scipy
import numpy as np
import argparse

TOTAL_PROCESSED = 0
TOTAL_LOOPS_CURRENT_RUN = 0

if os.path.exists('collect_apollo_saves.npy'):
    z = np.load('collect_apollo_saves.npy')
    TOTAL_PROCESSED = z[0]

def update_progress():
    np.save('collect_apollo_saves.npy', [TOTAL_PROCESSED])



def collect_for_camera(video_recording_dirs, camera_number):
    global TOTAL_PROCESSED
    global TOTAL_LOOPS_CURRENT_RUN
    print('checking for camera ' + camera_number)

    camera_folder = 'Camera '  + camera_number


    total_videos = len(video_recording_dirs)
    video_ite = 0
    for video_recording_dir in video_recording_dirs:
        video_ite = video_ite + 1
        print('----Processing video ' + str(video_ite) + ' of ' + str(total_videos)  )

        camera5_depth_folder_path = os.path.join(APOLLO_DATA_DEPTH_DIR, video_recording_dir, camera_folder)
        camera5_label_folder_path = os.path.join(APOLLO_DATA_LABEL_DIR, video_recording_dir, camera_folder)
        camera5_rgb_folder_path = os.path.join(APOLLO_DATA_RGB_DIR, video_recording_dir, camera_folder)

        if (not os.path.exists(camera5_depth_folder_path)) or (not os.path.exists(camera5_label_folder_path)) or (not os.path.exists(camera5_rgb_folder_path)):
            print('skipping missing folder: ' + video_recording_dir)
            continue

        images_list = os.listdir(camera5_depth_folder_path)
        images_list_size =  len(images_list)
        images_list_ite = 0
        for image_name in images_list:
            TOTAL_LOOPS_CURRENT_RUN = TOTAL_LOOPS_CURRENT_RUN + 1
            images_list_ite = images_list_ite + 1
            
            if TOTAL_LOOPS_CURRENT_RUN < TOTAL_PROCESSED:
                continue

            print('--------Processing Image ' + str(images_list_ite) + ' of ' + str(images_list_size) + '-------'  )
            
            out_filename1 = video_recording_dir + '_camera' + camera_number + '_' + image_name[0:-4] + '.npy'

            label_image_path = os.path.join(camera5_label_folder_path,image_name)
            if not os.path.exists(label_image_path):
                label_image_path = os.path.join(camera5_label_folder_path,image_name)[0:-4] + '_bin.png'
            
            if not os.path.exists(label_image_path):
                print('---------------Label not found ' + out_filename1)
                continue

            depth_img = misc.imread(os.path.join(camera5_depth_folder_path,image_name))
            label_img = misc.imread(label_image_path)
            rgb_img = misc.imread(os.path.join(camera5_rgb_folder_path,image_name)[0:-3] + 'jpg')

            #depth_img = Image.open(os.path.join(camera5_depth_folder_path,image_name))
            #label_img = Image.open(os.path.join(camera5_label_folder_path,image_name))
            #rgb_img = Image.open(os.path.join(camera5_rgb_folder_path,image_name)[0:-3] + 'jpg')

            #x = math.floor(depth_img.size[0] * IMG_RESIZE_RATIO)
            #y = math.floor(depth_img.size[1] * IMG_RESIZE_RATIO)

            depth_img = misc.imresize(depth_img, size=IMG_RESIZE_RATIO,interp='nearest', mode='F')
            label_img = misc.imresize(label_img, size=IMG_RESIZE_RATIO,interp='nearest')
            rgb_img = misc.imresize(rgb_img, size=IMG_RESIZE_RATIO,interp='nearest')



            x = depth_img.shape[0]
            y = depth_img.shape[1]
            output_array = np.empty((x,y,7))

            total_ignored = 0

            for x_ite in range(x):
                for y_ite in range(y):
                    # if x_ite % 200 == 0 and y_ite == 0:
                    #     print('%f done inserting', 100 * x_ite / x)
                    
                    output_array[x_ite][y_ite]=([
                        x_ite,
                        y_ite,
                        depth_img[x_ite][y_ite], 
                        rgb_img[x_ite][y_ite][0], 
                        rgb_img[x_ite][y_ite][1], 
                        rgb_img[x_ite][y_ite][2], 
                        label_img[x_ite][y_ite]
                    ])

            total = x * y
            print('total ignored: ', total_ignored, ' of ', total)

            data_label = np.concatenate(output_array, 0)



            np.save(os.path.join(OUTPUT_DIR,out_filename1), data_label)
            TOTAL_PROCESSED = TOTAL_PROCESSED + 1
            update_progress()
            
            


update_progress()

parser = argparse.ArgumentParser()
parser.add_argument('--abs_data_path', type=str, default='default', help='yes or no')
parser.add_argument('--abs_label_path', type=str, default='default', help='yes or no')
parser.add_argument('--abs_color_image_path', type=str, default='default', help='yes or no')
parser.add_argument('--abs_data_dir', type=str, default='default', help='yes or no')
FLAGS = parser.parse_args()


if FLAGS.abs_data_path == 'default':
    APOLLO_DATA_DEPTH_DIR = os.path.join(ROOT_DIR, 'data/annotation-apollo_scape_label-train_depth-apollo-1.5/depth')
else:
    APOLLO_DATA_DEPTH_DIR =  FLAGS.abs_data_path

if FLAGS.abs_label_path == 'default':
    APOLLO_DATA_LABEL_DIR = os.path.join(ROOT_DIR, 'data/annotation-apollo_scape_label-train_depth-apollo-1.5/label')
else:
    APOLLO_DATA_LABEL_DIR =  FLAGS.abs_label_path

if FLAGS.abs_color_image_path == 'default':
    APOLLO_DATA_RGB_DIR = os.path.join(ROOT_DIR, 'data/annotation-apollo_scape_label-train_depth-apollo-1.5/ColorImage')
else:
    APOLLO_DATA_RGB_DIR =  FLAGS.abs_color_image_path

if FLAGS.abs_data_dir == 'default':
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output/apollo_collected_data')
else:
    OUTPUT_DIR =  FLAGS.abs_data_dir
    
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

IMG_RESIZE_RATIO = 0.25

video_recording_dirs = [dir_name for dir_name in os.listdir(APOLLO_DATA_DEPTH_DIR) if os.path.isdir(os.path.join(APOLLO_DATA_DEPTH_DIR,dir_name))]
collect_for_camera(video_recording_dirs, '5')
video_recording_dirs = [dir_name for dir_name in os.listdir(APOLLO_DATA_DEPTH_DIR) if os.path.isdir(os.path.join(APOLLO_DATA_DEPTH_DIR,dir_name))]
collect_for_camera(video_recording_dirs, '6')
# for video_recording_dir in video_recording_dirs:
    


#     camera5_depth_folder_path = os.path.join(APOLLO_DATA_DEPTH_DIR, video_recording_dir, 'Camera 5')
#     camera5_label_folder_path = os.path.join(APOLLO_DATA_LABEL_DIR, video_recording_dir, 'Camera 5')
#     camera5_rgb_folder_path = os.path.join(APOLLO_DATA_RGB_DIR, video_recording_dir, 'Camera 5')
#     for image_name in os.listdir(camera5_depth_folder_path):
#         out_filename1 = video_recording_dir + '_camera5_' + image_name[0:-4] + '.npy'

#         depth_img = misc.imread(os.path.join(camera5_depth_folder_path,image_name))
#         label_img = misc.imread(os.path.join(camera5_label_folder_path,image_name))
#         rgb_img = misc.imread(os.path.join(camera5_rgb_folder_path,image_name)[0:-3] + 'jpg')

#         #depth_img = Image.open(os.path.join(camera5_depth_folder_path,image_name))
#         #label_img = Image.open(os.path.join(camera5_label_folder_path,image_name))
#         #rgb_img = Image.open(os.path.join(camera5_rgb_folder_path,image_name)[0:-3] + 'jpg')

#         #x = math.floor(depth_img.size[0] * IMG_RESIZE_RATIO)
#         #y = math.floor(depth_img.size[1] * IMG_RESIZE_RATIO)

#         depth_img = misc.imresize(depth_img, size=IMG_RESIZE_RATIO,interp='nearest', mode='F')
#         label_img = misc.imresize(label_img, size=IMG_RESIZE_RATIO,interp='nearest')
#         rgb_img = misc.imresize(rgb_img, size=IMG_RESIZE_RATIO,interp='nearest')



#         x = depth_img.shape[0]
#         y = depth_img.shape[1]
#         output_array = np.empty((x,y,7))

#         total_ignored = 0

#         for x_ite in range(x):
#             for y_ite in range(y):
#                 if x_ite % 200 == 0 and y_ite == 0:
#                     print('%f done inserting', 100 * x_ite / x)
                
#                 output_array[x_ite][y_ite]=([
#                     x_ite,
#                     y_ite,
#                     depth_img[x_ite][y_ite], 
#                     rgb_img[x_ite][y_ite][0], 
#                     rgb_img[x_ite][y_ite][1], 
#                     rgb_img[x_ite][y_ite][2], 
#                     label_img[x_ite][y_ite]
#                 ])

#         total = x * y
#         print('total ignored: ', total_ignored, ' of ', total)

#         data_label = np.concatenate(output_array, 0)



#         np.save(os.path.join(OUTPUT_DIR,out_filename1), data_label)
        
        


    