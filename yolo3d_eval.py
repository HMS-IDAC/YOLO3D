import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
import time
import importlib

from yolo3d_eval_params import *
from yolo3d_np_utils import load_data_noncontig, load_roi_data_noncontig, subsample_data_and_rois, \
    inference_on_large_volume
yolo_model = getattr(importlib.import_module('yolo3d_models'), model_name)


def eval_yolo():

    # print eval parameters
    with open('yolo3d_eval_params.py', 'r') as params_file:
        print('Eval parameters...')
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

    # load train/val/test image coords
    train_img_coords_list = pickle.load(open(os.path.join(model_dir, 'train_img_coords.p'), 'rb'))
    val_img_coords_list = pickle.load(open(os.path.join(model_dir, 'val_img_coords.p'), 'rb'))
    test_img_coords_list = pickle.load(open(os.path.join(model_dir, 'test_img_coords.p'), 'rb'))
    print('Train/val/test coordinates...')
    print('train_img_coords_list:' + str(train_img_coords_list))
    print('val_img_coords_list:' + str(val_img_coords_list))
    print('test_img_coords_list:' + str(test_img_coords_list))

    # load image volume.  store as 8bit to avoid memory problems
    print('loading data...')
    img_vol_list = load_data_noncontig(img_path_list)
    all_roi_data_list = load_roi_data_noncontig(roi_path_list)

    # split volumes into train/val/test
    print('splitting data into train/val/test...')
    train_img_vol_list, val_img_vol_list, test_img_vol_list = [], [], []
    train_roi_data_list, val_roi_data_list, test_roi_data_list = [], [], []
    num_train_rois_list, num_val_rois_list, num_test_rois_list = [], [], []
    for idx in range(len(all_roi_data_list)):
        # train
        train_img_vol, train_rois, _ = subsample_data_and_rois(img_vol_list[idx], all_roi_data_list[idx], [],
                                                               train_img_coords_list[idx], train_img_coords_list[idx])
        train_img_vol_list.append(train_img_vol)
        train_roi_data_list.append(train_rois)
        num_train_rois_list.append(len(train_rois))
        # val
        val_img_vol, val_rois, _ = subsample_data_and_rois(img_vol_list[idx], all_roi_data_list[idx], [],
                                                           val_img_coords_list[idx], val_img_coords_list[idx])
        val_img_vol_list.append(val_img_vol)
        val_roi_data_list.append(val_rois)
        num_val_rois_list.append(len(val_rois))
        # test
        test_img_vol, test_rois, _ = subsample_data_and_rois(img_vol_list[idx], all_roi_data_list[idx], [],
                                                             test_img_coords_list[idx], test_img_coords_list[idx])
        test_img_vol_list.append(test_img_vol)
        test_roi_data_list.append(test_rois)
        num_test_rois_list.append(len(test_rois))

    print('number of train rois = ' + str(np.sum(np.array(num_train_rois_list))))
    print('number of val rois = ' + str(np.sum(np.array(num_val_rois_list))))
    print('number of test rois = ' + str(np.sum(np.array(num_test_rois_list))))

    # delete original image volumes to save RAM
    del img_vol_list
    del all_roi_data_list

    # save train/val/test rois for quantitative evaluation
    pickle.dump(train_roi_data_list, open(os.path.join(out_dir, 'train_roi_data_list.p'), 'wb'))
    pickle.dump(val_roi_data_list, open(os.path.join(out_dir, 'val_roi_data_list.p'), 'wb'))
    pickle.dump(test_roi_data_list, open(os.path.join(out_dir, 'test_roi_data_list.p'), 'wb'))

    # INFERENCE ######################################################################################

    print('running inference on test volume...')
    start_time = time.time()

    all_pred_BBs_list = []
    all_confs_list = []
    for idx in range(len(test_img_vol_list)):
        img_vol = test_img_vol_list[idx]
        all_pred_BBs, all_confs = inference_on_large_volume(sess, pred, tf_data, keep_prob, img_vol, conf_thresh,
                                                            eval_batch_size, imsize, s, num_classes)
        all_pred_BBs_list.append(all_pred_BBs)
        all_confs_list.append(all_confs)

    elapsed_time = time.time() - start_time
    print("elapsed time for inference on test volume = " + str(elapsed_time) + " seconds")

    pickle.dump(zip(all_pred_BBs_list, all_confs_list), open(os.path.join(out_dir, 'predictions_test.p'), 'wb'))

    print('running inference on val volume...')
    start_time = time.time()

    all_pred_BBs_list = []
    all_confs_list = []
    for idx in range(len(val_img_vol_list)):
        img_vol = val_img_vol_list[idx]
        all_pred_BBs, all_confs = inference_on_large_volume(sess, pred, tf_data, keep_prob, img_vol, conf_thresh,
                                                            eval_batch_size, imsize, s, num_classes)
        all_pred_BBs_list.append(all_pred_BBs)
        all_confs_list.append(all_confs)

    elapsed_time = time.time() - start_time
    print("elapsed time for inference on val volume = " + str(elapsed_time) + " seconds")

    pickle.dump(zip(all_pred_BBs_list, all_confs_list), open(os.path.join(out_dir, 'predictions_val.p'), 'wb'))

    print('running inference on train volume...')
    start_time = time.time()

    all_pred_BBs_list = []
    all_confs_list = []
    for idx in range(len(train_img_vol_list)):
        img_vol = train_img_vol_list[idx]
        all_pred_BBs, all_confs = inference_on_large_volume(sess, pred, tf_data, keep_prob, img_vol, conf_thresh,
                                                            eval_batch_size, imsize, s, num_classes)
        all_pred_BBs_list.append(all_pred_BBs)
        all_confs_list.append(all_confs)

    elapsed_time = time.time() - start_time
    print("elapsed time for inference on train volume = " + str(elapsed_time) + " seconds")

    pickle.dump(zip(all_pred_BBs_list, all_confs_list), open(os.path.join(out_dir, 'predictions_train.p'), 'wb'))

    sess.close()

def main():

    # run eval
    eval_yolo()


if __name__ == '__main__':
    main()
