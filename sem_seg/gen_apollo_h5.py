import os
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import data_prep_util
import indoor3d_util
from random import shuffle

# Constants
data_dir = os.path.join(ROOT_DIR, 'data')
#indoor3d_data_dir = os.path.join(data_dir, 'stanford_indoor3d')
NUM_POINT = 4096
H5_BATCH_SIZE = 100
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

data_label_file_path = os.path.join(ROOT_DIR, 'output/apollo_collected_data')
# Set paths
#filelist = os.path.join(BASE_DIR, 'meta/all_data_label.txt')
#data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]
APOLLO_DATA_DEPTH_DIR = os.path.join(ROOT_DIR, 'data/test-apollo/depth')
video_recording_dirs = [os.path.join(APOLLO_DATA_DEPTH_DIR, dir_name) for dir_name in os.listdir(APOLLO_DATA_DEPTH_DIR) if os.path.isdir(os.path.join(APOLLO_DATA_DEPTH_DIR,dir_name))]

#data_label_files = []
#for video_dir in video_recording_dirs:
#    y = [os.path.join(video_dir,  image_name) for image_name in os.listdir(video_dir)]
#    data_label_files += y

data_label_files = [os.path.join(data_label_file_path, data_file) for data_file in os.listdir(data_label_file_path) ]


class_to_train_mappings =  [[int(line.rstrip().split(',')[0]),int(line.rstrip().split(',')[1])] for line in open('data/annotation-apollo_scape_label-train_depth-apollo-1.5/class-to-train-mappings.txt')]



    


shuffle(data_label_files)


# output_dir = os.path.join(data_dir, 'apollo_sem_seg_hdf5_data')
output_dir = os.path.join(data_dir, 'apollo_sem_seg_hdf5_data_test')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')
fout_labels = open(os.path.join(output_dir, 'class_mappings.txt'), 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...] 
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index).zfill(6) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype) 
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index).zfill(6) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0

label_selections = [
        33,
        161,

        #36,
        #164,

        #39,
        #167,

        113,



        49,

        84,

        97

    ]


select_class_id_mappings = np.copy(label_selections)

for from_number,to_number in class_to_train_mappings:
    select_class_id_mappings[select_class_id_mappings == from_number] = to_number

select_class_id_mappings = np.unique(select_class_id_mappings)
select_class_id_mappings.sort()
for class_id in select_class_id_mappings:
    fout_labels.write(str(class_id) + '\n')
fout_labels.close()


selected_class_id_to_train_id_conversion = []

for from_number,to_number in class_to_train_mappings:
    if to_number in select_class_id_mappings:
        selected_class_id_to_train_id_conversion.append([from_number, np.where(select_class_id_mappings == to_number)[0][0]])


total_seen_class = [0 for _ in range(len(select_class_id_mappings))]



#for i, data_label_filename in enumerate(data_label_files):
for i, data_label_filename in enumerate(data_label_files):
    
    #print(data_label_filename)
    data, label = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=256.0, stride=128,
                                                 random_sample=False, sample_num=None, label_selections=label_selections)

    
    
    for from_val, to_val in selected_class_id_to_train_id_conversion:
        idxs = label == from_val
        label[idxs] = to_val
        
        total_seen_class[to_val] += np.sum(idxs)

    print('{0}, {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]

    if i > 65:
        insert_batch(data, label, True)
        break

    insert_batch(data, label, i == len(data_label_files)-1)

fout_room.close()
print("Total samples: {0}".format(sample_cnt))
print("total instaesnc: ")
print(total_seen_class)