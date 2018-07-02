import math
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 12]')
parser.add_argument('--learning_rate', type=float, default=0.000001, help='Initial learning rate [default: 0.000001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='momentum', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_recordings', type=str, default='11', help='Which recording numbers to use for test, i.e "1,2", "1", "3", "3,4,5" [default: 11]')
parser.add_argument('--dir_path_h5', type=str, default='data/apollo_sem_seg_hdf5_data_test', help='directory containing the h5 files')
parser.add_argument('--use_saved_model', type=str, default='no', help='yes or no')

FLAGS = parser.parse_args()

LOAD_FULL_DATA  = False
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)

USE_SAVED_MODEL = False
if FLAGS.use_saved_model == 'yes':
    USE_SAVED_MODEL = True
    print('using saved model')
elif FLAGS.use_saved_model != 'no':
    raise ValueError('use_saved_model param must be eitehr yes or no')


os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# DIR_PATH_H5 = os.path.join(ROOT_DIR, 'data/apollo_sem_seg_hdf5_data_test')
DIR_PATH_H5 = FLAGS.dir_path_h5
if not os.path.exists(DIR_PATH_H5):
    raise ValueError('the given h5 directory is invalid')
H5_FILES = [os.path.join(DIR_PATH_H5, file_h5) for file_h5 in os.listdir(DIR_PATH_H5) if file_h5[-2:] == 'h5']


#ALL_FILES = provider.getDataFiles('data/apollo_sem_seg_hdf5_data')
room_filelist = [line.rstrip() for line in open(os.path.join(DIR_PATH_H5, 'room_filelist.txt'))]
classMappings = [line.rstrip() for line in open(os.path.join(DIR_PATH_H5, 'class_mappings.txt'))]
NUM_CLASSES = len(classMappings)



BATCH_SIZE_H5 = provider.loadDataFile(H5_FILES[0])[0].shape[0]

# Load ALL data

# if LOAD_FULL_DATA: 
#     data_batch_list = []
#     label_batch_list = []
#     for i,h5_filename in enumerate(H5_FILES):
#         if i%10 == 0:
#             print("loading h5 file: " , i, h5_filename)
        
#         data_batch, label_batch = provider.loadDataFile(h5_filename)
#         data_batch_list.append(data_batch)
#         label_batch_list.append(label_batch)



# if LOAD_FULL_DATA: 
#     print('---all loaded---')
#     data_batches = np.concatenate(data_batch_list, 0)
#     data_batch_list = None
#     label_batches = np.concatenate(label_batch_list, 0)
#     label_batch_list = None
#     print(data_batches.shape)
#     print(label_batches.shape)


data_for_training = np.empty(len(room_filelist), dtype=bool)


test_recordings = [str(int(recording_number)).zfill(3) for recording_number in FLAGS.test_recordings.split(',')]
#test_recordings = 'Area_'+str(FLAGS.test_area)


# if LOAD_FULL_DATA: 
#     train_idxs = []
#     test_idxs = [] 

total_training_data = 0
total_testing_data = 0
for i,room_name in enumerate(room_filelist):

    #remove this
    if i%4==0:
        total_testing_data += 1
        data_for_training[i] = False
    #if room_name[6:9] in test_recordings:


        # if LOAD_FULL_DATA: 
        #     test_idxs.append(i)
    else:
        total_training_data += 1
        data_for_training[i] = True


        # if LOAD_FULL_DATA: 
        #     train_idxs.append(i)


# if LOAD_FULL_DATA: 
#     train_data = data_batches[train_idxs,...]
#     train_label = label_batches[train_idxs]
#     test_data = data_batches[test_idxs,...]
#     test_label = label_batches[test_idxs]
#     data_batches = None
#     label_batches = None
#     print(train_data.shape, train_label.shape)
#     print(test_data.shape, test_label.shape)


current_train_idx = 0
current_test_idx = 0
last_loaded_file_index = None
last_loaded_file_data = None
last_loaded_file_label = None
def reset_train_data():
    global current_train_idx
    current_train_idx = 0

def reset_test_data():
    global current_test_idx
    current_test_idx = 0

def can_get_test_data():

    global current_test_idx
    return current_test_idx <  data_for_training.shape[0]
def can_get_train_data():
    global current_train_idx
    global last_loaded_file_index
    global last_loaded_file_data
    global last_loaded_file_label

    return current_train_idx <  data_for_training.shape[0]

    # h5_fileindex = int(math.floor( current_train_idx / float(BATCH_SIZE_H5) ))

    # if h5_fileindex + 1 < len(H5_FILES):
    #     return True
    
    # if last_loaded_file_index != h5_fileindex:
    #     h5_filename = H5_FILES[h5_fileindex]
    #     last_loaded_file_data, last_loaded_file_label = provider.loadDataFile(h5_filename)
    #     last_loaded_file_index = h5_fileindex



    # start_idx_batch = current_train_idx - (h5_fileindex * BATCH_SIZE_H5)
 
    # h5_remaining_batch_size = BATCH_SIZE_H5 - start_idx_batch

    # return h5_remaining_batch_size > 0



def get_train_or_test_data(amount, for_training):
    global current_train_idx
    global current_test_idx

    global last_loaded_file_index
    global last_loaded_file_data
    global last_loaded_file_label

    local_data_batch_list = []
    local_label_batch_list = []

    total_retrieved = 0

    if for_training:
        index_for_run = current_train_idx
    else:            
        index_for_run = current_test_idx


    while total_retrieved < amount and index_for_run <  data_for_training.shape[0]:
     

        #total_retrieved += 1

        h5_fileindex = int(math.floor( index_for_run / float(BATCH_SIZE_H5) ))


        if last_loaded_file_index != h5_fileindex:
            h5_filename = H5_FILES[h5_fileindex]
            last_loaded_file_data, last_loaded_file_label = provider.loadDataFile(h5_filename)
            last_loaded_file_index = h5_fileindex


        amount_to_retrieve = amount - total_retrieved

        start_idx_batch = index_for_run - (h5_fileindex * BATCH_SIZE_H5)

        h5_remaining_batch_size = BATCH_SIZE_H5 - start_idx_batch

        amount_to_fetch_from_batch = min(amount_to_retrieve, h5_remaining_batch_size)

        start_idx_total = index_for_run
        end_idx_total = start_idx_total + amount_to_fetch_from_batch

        
        end_idx_batch = start_idx_batch + amount_to_fetch_from_batch 


        if for_training:
            data_batch = (last_loaded_file_data[start_idx_batch:end_idx_batch]) [data_for_training[start_idx_total:end_idx_total],:,:]
            label_batch = (last_loaded_file_label[start_idx_batch:end_idx_batch]) [data_for_training[start_idx_total:end_idx_total],:]
        else:
            arr = data_for_training[start_idx_total:end_idx_total] == False

            data_batch = (last_loaded_file_data[start_idx_batch:end_idx_batch]) [arr,:,:]
            label_batch = (last_loaded_file_label[start_idx_batch:end_idx_batch]) [arr,:]

        total_retrieved += data_batch.shape[0]
        index_for_run += amount_to_fetch_from_batch

        local_data_batch_list.append(data_batch)
        local_label_batch_list.append(label_batch)

    local_data_batches = np.concatenate(local_data_batch_list, 0)
    local_label_batches = np.concatenate(local_label_batch_list, 0)


    if for_training:
        current_train_idx = index_for_run
    else:
        current_test_idx = index_for_run

    return local_data_batches, local_label_batches


    



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train(use_saved_model ):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, num_classes=NUM_CLASSES)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)



        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})


        if use_saved_model:
            saver.restore(sess, os.path.join(LOG_DIR,'model.ckpt'))

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))


        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}


        if use_saved_model:
            eval_one_epoch(sess, ops, test_writer)
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 1 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    reset_train_data()
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')

    #checking to confirm get_train_data is functioning correctly

    # if LOAD_FULL_DATA: 
    #     current_data = train_data
    #     current_label = train_label
    #     file_size = current_data.shape[0]
    #     num_batches = file_size // BATCH_SIZE
    #     num_batches = total_training_data / BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    batch_idx = -1
    # for batch_idx in range(num_batches):
    while can_get_train_data():
        batch_idx += 1
        
        if batch_idx % 10 == 0:
            print('Current batch: %d'%(batch_idx))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        if batch_idx == 118:
            z=123123

        data_for_loop, label_for_loop = get_train_or_test_data(BATCH_SIZE, True)

        #this is in case the last batch has insufficient blocks, so we simply bail
        if not can_get_train_data():
            break;    

        #checking to confirm get_train_data is functioning correctly
        # check_data_for_loop = current_data[start_idx:end_idx, :, :]
        # check_label_for_loop = current_label[start_idx:end_idx]
        # if sum(sum(sum(data_for_loop == check_data_for_loop))) != 442368:
        #     z = 32131
        #     log_string('check data for loop not match what it should be')
        #     raise ValueError('check data for loop not match what it should be')
        

        #remove below comments
        
        feed_dict = {ops['pointclouds_pl']: data_for_loop,
                     ops['labels_pl']: label_for_loop,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == label_for_loop)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val


    #remove below comments
    
    # log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    # log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    reset_test_data()
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    # current_data = test_data[:,0:NUM_POINT,:]
    # current_label = np.squeeze(test_label)
    
    # file_size = current_data.shape[0]
    # num_batches = file_size // BATCH_SIZE
    
    batch_idx = -1
    # for batch_idx in range(num_batches):
    while can_get_test_data():
        batch_idx += 1


        data_for_loop, label_for_loop = get_train_or_test_data(BATCH_SIZE, False)

        #this is in case the last batch has insufficient blocks
        if not can_get_test_data():
            break

        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: data_for_loop,
                     ops['labels_pl']: label_for_loop,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == label_for_loop)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                try: 
                    l = label_for_loop[i - start_idx, j - start_idx]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i-start_idx, j] == l)
                except: 
                    l = label_for_loop[i - start_idx, j - start_idx]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i-start_idx, j] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    print('total correct class')
    print(total_correct_class)
    print('total seen class')
    print(total_seen_class)


if __name__ == "__main__":
    train(USE_SAVED_MODEL)
    LOG_FOUT.close()
