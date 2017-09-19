import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
import importlib

from yolo3d_train_params import *
from yolo3d_np_utils import load_data_noncontig, load_roi_data_noncontig, subsample_data_and_rois, \
    sample_pos_batch_noncontig, sample_neg_batch_noncontig, preprocess_image
from yolo3d_tf_utils import yolo_loss
yolo_model = getattr(importlib.import_module('yolo3d_models'), model_name)


def train_yolo():

    # print training parameters
    with open('yolo3d_train_params.py', 'r') as params_file:
        print('Training parameters...')
        print(params_file.read())

    # GRAPH CONSTRUCTION #######################################################################

    # initialize yolo model
    tf_data = tf.placeholder(tf.float32, shape=[None, imsize[0], imsize[1], imsize[2], nchannels])
    labels = tf.placeholder(tf.float32, shape=(None, s * s * s * (7 + num_classes)))
    keep_prob = tf.placeholder(tf.float32)

    pred = yolo_model(tf_data, s, num_bb, num_classes, keep_prob)
    loss = yolo_loss(pred, labels, s, num_bb, num_classes)

    # define optimization op
    train_step = tf.train.AdamOptimizer(base_lr).minimize(loss)

    # LOGS #####################################################################################

    # specify what you want to log
    tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()

    # create summary writer object
    train_writer = tf.summary.FileWriter(out_dir + '/train')  # ,graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(out_dir + '/val')  # ,graph=tf.get_default_graph())

    # LAUNCH SESSION ###########################################################################

    # specify gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # launch session
    sess = tf.Session()

    # create saver object.  saver needs to be in cpu scope, b/c writes to disk
    with tf.device('cpu:0'):
        saver = tf.train.Saver()  # max_to_keep=None)

    # initialize all variables
    if init_model == 'random':
        # init from random
        sess.run(tf.global_variables_initializer())
    else:
        # init from previous training session
        saver.restore(sess, os.path.join(init_model, 'iter-' + str(init_iters)))

    # LOAD DATA ################################################################################

    # load image volume.  store as 8bit to avoid memory problems
    print('loading data...')
    img_vol_list = load_data_noncontig(img_path_list)
    all_roi_data_list = load_roi_data_noncontig(roi_path_list)

    # split volumes into train/val to avoid cross-talk of negative samples
    print('splitting data into train/val...')
    train_img_vol_list, val_img_vol_list = [], []
    train_roi_data_list, val_roi_data_list = [], []
    num_train_rois_list, num_val_rois_list = [], []
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

    print('number of train rois = ' + str(np.sum(np.array(num_train_rois_list))))
    print('number of val rois = ' + str(np.sum(np.array(num_val_rois_list))))

    # delete original image volumes to save RAM
    del img_vol_list
    del all_roi_data_list

    # save train/val/test for evaluation
    pickle.dump(train_img_coords_list, open(os.path.join(out_dir, 'train_img_coords.p'), 'wb'))
    pickle.dump(val_img_coords_list, open(os.path.join(out_dir, 'val_img_coords.p'), 'wb'))
    pickle.dump(test_img_coords_list, open(os.path.join(out_dir, 'test_img_coords.p'), 'wb'))

    # for simplicity
    num_pos_train = int(frac_pos * train_batch_size)
    num_neg_train = train_batch_size - num_pos_train
    num_pos_val = int(frac_pos * val_batch_size)
    num_neg_val = val_batch_size - num_pos_val

    # TRAIN ######################################################################################

    print('starting training...')
    for i_iter in range(n_iters):

        # load batch and pre-process
        pos_batch_imgs, pos_batch_labels = sample_pos_batch_noncontig(train_img_vol_list, train_roi_data_list,
                                                                      num_pos_train, s, num_classes)
        neg_batch_imgs, neg_batch_labels = sample_neg_batch_noncontig(train_img_vol_list, train_roi_data_list,
                                                                      num_neg_train, s, num_classes)
        batch_imgs = np.concatenate((pos_batch_imgs, neg_batch_imgs))
        batch_labels = np.concatenate((pos_batch_labels, neg_batch_labels))

        batch_imgs = preprocess_image(batch_imgs)

        sess.run(train_step, feed_dict={tf_data: batch_imgs, labels: batch_labels, keep_prob: dropout_keep_prob})

        # train loss
        if i_iter % log_interval == 0:
            # resample train batch.  Takes a bit of time, but more accurate loss
            pos_batch_imgs, pos_batch_labels = sample_pos_batch_noncontig(train_img_vol_list, train_roi_data_list,
                                                                          num_pos_train, s, num_classes)
            neg_batch_imgs, neg_batch_labels = sample_neg_batch_noncontig(train_img_vol_list, train_roi_data_list,
                                                                          num_neg_train, s, num_classes)
            batch_imgs = np.concatenate((pos_batch_imgs, neg_batch_imgs))
            batch_labels = np.concatenate((pos_batch_labels, neg_batch_labels))

            batch_imgs = preprocess_image(batch_imgs)

            # run ops
            [summary, train_loss] = sess.run([summary_op, loss],
                                             feed_dict={tf_data: batch_imgs, labels: batch_labels, keep_prob: 1.})
            train_writer.add_summary(summary, i_iter)
            print("step %d, train loss %g" % (i_iter, train_loss))

        # val loss
        if i_iter % val_interval == 0:
            # sample from val set
            pos_batch_imgs, pos_batch_labels = sample_pos_batch_noncontig(val_img_vol_list, val_roi_data_list,
                                                                          num_pos_val, s, num_classes)
            neg_batch_imgs, neg_batch_labels = sample_neg_batch_noncontig(val_img_vol_list, val_roi_data_list,
                                                                          num_neg_val, s, num_classes)
            batch_imgs = np.concatenate((pos_batch_imgs, neg_batch_imgs))
            batch_labels = np.concatenate((pos_batch_labels, neg_batch_labels))

            batch_imgs = preprocess_image(batch_imgs)

            # run ops
            [summary, val_loss] = sess.run([summary_op, loss],
                                           feed_dict={tf_data: batch_imgs, labels: batch_labels, keep_prob: 1.})
            val_writer.add_summary(summary, i_iter)
            print("step %d, val loss %g" % (i_iter, val_loss))

        if i_iter % save_interval == 0:
            out_file = out_dir + os.path.sep + 'iter'
            save_path = saver.save(sess, out_file, global_step=i_iter)
            print("Model checkpoint saved to file: " + save_path)

    sess.close()


def main():
    # run train
    train_yolo()

if __name__ == '__main__':
    main()
