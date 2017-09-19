import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
import time
import importlib

from yolo3d_run_params import *
from yolo3d_np_utils import load_data_noncontig, random_batch, decode_prediction
yolo_model = getattr(importlib.import_module('yolo3d_models'), model_name)


def run_yolo():

    # print eval parameters
    with open('yolo3d_run_params.py', 'r') as params_file:
        print('Run parameters...')
        print(params_file.read())

    # GRAPH CONSTRUCTION #######################################################################

    # create model
    tf_data = tf.placeholder(tf.float32, shape=[None, imsize[0], imsize[1], imsize[2], nchannels])
    keep_prob = tf.placeholder(tf.float32)
    pred = yolo_model(tf_data, s, num_bb, num_classes, keep_prob)

    # LAUNCH SESSION ###########################################################################

    # specify gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # launch session
    sess = tf.Session()

    with tf.device('cpu:0'):
        saver = tf.train.Saver()

    saver.restore(sess, os.path.join(model_dir, 'iter-' + str(iter_num)))

    # LOAD DATA ################################################################################

    # load full image volume.  store as 8bit to avoid memory problems
    print('loading data...')
    img_vol_list = load_data_noncontig(img_path_list)
    img_vol = img_vol_list[0]

    # crop random positions from within image volume (xx: this should be replaced by sampling positions you care about)
    batch_imgs, coords = random_batch(img_vol, batch_size, imsize)

    # INFERENCE #################################################################################

    output = sess.run(pred, feed_dict={tf_data: batch_imgs, keep_prob: 1.})

    # PARSE OUTPUT ##############################################################################

    # convert prediction to: [class,[x,y,z,w,h,t]]
    all_pred_BBs = []
    all_confs = []
    for i in range(batch_size):
        pred_BBs, confs = decode_prediction(output[i, :], imsize, s, num_classes, conf_thresh)

        for pred_num in range(len(pred_BBs)):
            pred_BB_label = pred_BBs[pred_num][0]
            pred_BB_coords = np.array(pred_BBs[pred_num][1])
            pred_BB_coords[0:3] += np.array(coords[i])
            all_pred_BBs.append([pred_BB_label, pred_BB_coords])
            all_confs.append(confs[pred_num])

    # SAVE TO FILE ##############################################################################

    pickle.dump(zip(all_pred_BBs, all_confs), open(os.path.join(out_dir, 'predictions.p'), 'wb'))

    sess.close()


def main():

    # run eval
    run_yolo()


if __name__ == '__main__':
    main()
