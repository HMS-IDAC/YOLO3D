import numpy as np
import myijroi
import os
import random
import re
import skimage.io
import skimage.filters
from PIL import Image, ImageDraw


def load_data(img_path, preprocess_flag=0):
    #
    img_vol = []
    for img_filename in os.listdir(img_path):
        if img_filename.endswith('.tif'):
            img_vol.append(skimage.io.imread(os.path.join(img_path, img_filename)))
    img_vol = np.stack(img_vol)

    if preprocess_flag == 1:
        img_vol = preprocess_image(img_vol)

    return img_vol


def load_data_noncontig(img_path_list, preprocess_flag=0):
    #
    img_vol_list = []
    for img_path in img_path_list:
        img_vol_list.append(load_data(img_path, preprocess_flag))

    return img_vol_list


def preprocess_image(img):
    # scale to [-0.5, 0.5)
    img = (img - 128.) * 0.00390625

    return img


def unprocess_image(img):
    # scale to [0,256)
    img = img / 0.00390625 + 128.

    return img


def load_roi_data(roi_root, label_names=None):
    # note this is currently identical to 2D

    zips_list = [f for f in os.listdir(roi_root) if f.endswith('.zip')]
    all_roi_data = []

    for zip_file in zips_list:

        # info that's common to all rois in zip file ----------------------------------------------------------

        # slice offset
        first_slice = re.findall('slices=(.+)-', zip_file)
        first_slice = int(first_slice[0])

        # open zip --------------------------------------------------------------------------------------------

        # load all rois, and store in tuple (label,(slice,x,y,w,h))
        roi_path = os.path.join(roi_root, zip_file)
        with open(roi_path, 'rb') as f:
            roi_list = myijroi.read_roi_zip(f)

        for roi in roi_list:
            # parse tuple
            roi_name = roi[0]
            roi_BB = roi[1]

            # calculate ROI params
            x = (roi_BB[0, 1] + roi_BB[1, 1]) / 2
            y = (roi_BB[1, 0] + roi_BB[2, 0]) / 2
            w = roi_BB[1, 1] - roi_BB[0, 1]
            h = roi_BB[2, 0] - roi_BB[1, 0]
            slice_num = roi_BB[4, 0]

            # slice
            slice_num += first_slice - 1
            z_num = slice_num - 1  # zero indexing into image volume for python

            # extract label
            label = 0   # default
            if label_names:
                for idx, label_name in enumerate(label_names):
                    if re.findall('(.+) \d', roi_name)[0] == label_name:
                        label = idx
                        break

            # store all
            roi_data = (label, (z_num, x, y, w, h))
            all_roi_data.append(roi_data)

    return all_roi_data


def load_roi_data_noncontig(roi_path_list):
    all_roi_data_list = []

    for roi_path in roi_path_list:
        all_roi_data_list.append(load_roi_data(roi_path))

    return all_roi_data_list


def convert_roidata_roicw(roi_data, z_calibration=0.1):

    z = roi_data[1][0]
    x = roi_data[1][1]
    y = roi_data[1][2]
    w = roi_data[1][3]
    h = roi_data[1][4]

    # estimate thickness from width and height
    t = (w + h) / 2.
    t = np.round(t * z_calibration)
    t = np.int32(t)

    roi_cw = [x, y, z, w, h, t]

    return roi_cw


def convert_cw_lr(roi_cw):
    x1 = roi_cw[0] - roi_cw[3] / 2
    x2 = roi_cw[0] + roi_cw[3] / 2
    y1 = roi_cw[1] - roi_cw[4] / 2
    y2 = roi_cw[1] + roi_cw[4] / 2
    z1 = roi_cw[2] - roi_cw[5] / 2
    z2 = roi_cw[2] + roi_cw[5] / 2

    roi_lr = [x1, x2, y1, y2, z1, z2]

    return roi_lr


def convert_lr_cw(roi_lr):
    x = (roi_lr[0] + roi_lr[1]) / 2
    y = (roi_lr[2] + roi_lr[3]) / 2
    z = (roi_lr[4] + roi_lr[5]) / 2
    w = roi_lr[1] - roi_lr[0]
    h = roi_lr[3] - roi_lr[2]
    t = roi_lr[5] - roi_lr[4]

    roi_cw = [x, y, z, w, h, t]

    return roi_cw


def iou_np(box1_cw, box2_cw):
    box1_lr = convert_cw_lr(box1_cw)
    box2_lr = convert_cw_lr(box2_cw)

    x_int = np.float32(np.minimum(box1_lr[1], box2_lr[1])) - np.maximum(box1_lr[0], box2_lr[0])
    y_int = np.float32(np.minimum(box1_lr[3], box2_lr[3])) - np.maximum(box1_lr[2], box2_lr[2])
    z_int = np.float32(np.minimum(box1_lr[5], box2_lr[5])) - np.maximum(box1_lr[4], box2_lr[4])

    if (x_int <= 0) or (y_int <= 0) or (z_int <= 0):
        intersection = np.float32(0)
    else:
        intersection = x_int * y_int * z_int

    vol1 = np.float32(box1_cw[3]) * box1_cw[4] * box1_cw[5]
    vol2 = np.float32(box2_cw[3]) * box2_cw[4] * box2_cw[5]

    return intersection / (vol1 + vol2 - intersection)


def iou_np_fast(box1_cw, box2_cw):
    box1_lr = convert_cw_lr(box1_cw)
    box2_lr = convert_cw_lr(box2_cw)

    x_int = np.minimum(box1_lr[1], box2_lr[1]) - np.maximum(box1_lr[0], box2_lr[0])
    if x_int <= 0:
        return 0

    y_int = np.minimum(box1_lr[3], box2_lr[3]) - np.maximum(box1_lr[2], box2_lr[2])
    if y_int <= 0:
        return 0

    z_int = np.minimum(box1_lr[5], box2_lr[5]) - np.maximum(box1_lr[4], box2_lr[4])
    if z_int <= 0:
        return 0

    intersection = np.float32(x_int) * y_int * z_int

    vol1 = np.float32(box1_cw[3]) * box1_cw[4] * box1_cw[5]
    vol2 = np.float32(box2_cw[3]) * box2_cw[4] * box2_cw[5]

    return np.float32(intersection) / (vol1 + vol2 - intersection)


def sample_pos_window(pos_roi_cw, im_shape, window_shape):
    # mvb bb
    [x1_mvb, x2_mvb, y1_mvb, y2_mvb, z1_mvb, z2_mvb] = convert_cw_lr(pos_roi_cw)

    # range of positions where I can crop volume, such that mvb is inside
    x1_range1 = x2_mvb - window_shape[2]
    x1_range2 = x1_mvb

    y1_range1 = y2_mvb - window_shape[1]
    y1_range2 = y1_mvb

    z1_range1 = z2_mvb - window_shape[0]
    z1_range2 = z1_mvb

    # random offset
    x1 = np.random.rand() * (x1_range2 - x1_range1) + x1_range1
    x1 = np.int32(x1)
    x2 = x1 + window_shape[2]

    y1 = np.random.rand() * (y1_range2 - y1_range1) + y1_range1
    y1 = np.int32(y1)
    y2 = y1 + window_shape[1]

    z1 = np.random.rand() * (z1_range2 - z1_range1) + z1_range1
    z1 = np.int32(z1)
    z2 = z1 + window_shape[0]

    # take care of out-of-bounds
    if x1 < 0:
        x1 = 0
        x2 = window_shape[2]
    elif x2 > im_shape[2]:
        x2 = im_shape[2]
        x1 = x2 - window_shape[2]

    if y1 < 0:
        y1 = 0
        y2 = window_shape[1]
    elif y2 > im_shape[1]:
        y2 = im_shape[1]
        y1 = y2 - window_shape[1]

    if z1 < 0:
        z1 = 0
        z2 = window_shape[0]
    elif z2 > im_shape[0]:
        z2 = im_shape[0]
        z1 = z2 - window_shape[0]

    return [x1, x2, y1, y2, z1, z2]


def sample_neg_window(img_vol_shape, all_roi_data, crop_shape=[32, 256, 256]):
    while True:

        # randomly sample coords of window
        x1 = np.random.randint(img_vol_shape[2] - crop_shape[2])
        x2 = x1 + crop_shape[2]

        y1 = np.random.randint(img_vol_shape[1] - crop_shape[1])
        y2 = y1 + crop_shape[1]

        z1 = np.random.randint(img_vol_shape[0] - crop_shape[0])
        z2 = z1 + crop_shape[0]

        # convert to cw format, for iou with rois
        crop_lr = [x1, x2, y1, y2, z1, z2]
        crop_cw = convert_lr_cw(crop_lr)

        for roi_data in all_roi_data:

            roi_cw = convert_roidata_roicw(roi_data)
            iou = iou_np(crop_cw, roi_cw)

            if iou > 0:
                break

        if iou == 0:
            break

    return crop_lr


def add_bbs_to_window(img_coords_lr, pos_roi_data, other_rois_data):
    # pull out xyzwht from pos MVB
    pos_roi_cw = convert_roidata_roicw(pos_roi_data)

    other_bbs = []

    for other_roi_data in other_rois_data:

        # unpack roi
        other_roi_cw = convert_roidata_roicw(other_roi_data)
        label = other_roi_data[0]

        # check if roi is in window ---------------------------------------------------

        # require that center of other_roi is in the image volume
        if other_roi_cw[0] < img_coords_lr[0]: continue
        if other_roi_cw[0] >= img_coords_lr[1]: continue
        if other_roi_cw[1] < img_coords_lr[2]: continue
        if other_roi_cw[1] >= img_coords_lr[3]: continue
        if other_roi_cw[2] < img_coords_lr[4]: continue
        if other_roi_cw[2] >= img_coords_lr[5]: continue

        # finally, check if roi overlaps with previous rois ---------------------------

        # check if roi is the same as the original roi (but labeled in another slice)
        max_iou = iou_np(other_roi_cw, pos_roi_cw)

        # check if the roi overlaps with previously added 'other' rois
        for other_bb in other_bbs:
            other_iou = iou_np(other_roi_cw, other_bb[1])
            if other_iou > max_iou:
                max_iou = other_iou

        if max_iou > 0.25:
            continue

        other_bbs.append([label, np.asarray(other_roi_cw)])

    return other_bbs


def encode_pos_bbs(pos_bbs, im_shape, s, num_classes):

    enc_class = np.zeros((s * s * s, num_classes))
    enc_bb = np.zeros((s * s * s, 7))

    for pos_bb in pos_bbs:

        label = pos_bb[0]
        pos_bb_coords = pos_bb[1]

        grid_size_x = im_shape[2] / s
        grid_size_y = im_shape[1] / s
        grid_size_z = im_shape[0] / s

        grid_x = pos_bb_coords[0] // grid_size_x
        grid_y = pos_bb_coords[1] // grid_size_y
        grid_z = pos_bb_coords[2] // grid_size_z

        grid_idx = grid_z * s * s + grid_y * s + grid_x

        x = np.float32(pos_bb_coords[0] % grid_size_x) / grid_size_x
        y = np.float32(pos_bb_coords[1] % grid_size_y) / grid_size_y
        z = np.float32(pos_bb_coords[2] % grid_size_z) / grid_size_z

        w = np.float32(pos_bb_coords[3]) / grid_size_x
        h = np.float32(pos_bb_coords[4]) / grid_size_y
        t = np.float32(pos_bb_coords[5]) / grid_size_z

        # encode
        enc_class[grid_idx, label] = 1
        enc_bb[grid_idx, :] = np.array([1., x, y, z, w, h, t])

    enc = np.concatenate((np.reshape(enc_class, (s * s * s * num_classes)), np.reshape(enc_bb, (s * s * s * 7))))

    return enc


def decode_labels(enc_labels, im_shape, s, num_classes):

    enc_class = np.reshape(enc_labels[:s * s * s * num_classes], (s * s * s, num_classes))
    enc_bb = np.reshape(enc_labels[s * s * s * num_classes:], (s * s * s, 7))

    pos_bbs = []
    for grid_idx in range(s * s * s):

        if enc_bb[grid_idx, 0] == 0:
            continue

        grid_x = (grid_idx % (s * s)) % s
        grid_y = (grid_idx % (s * s)) // s
        grid_z = grid_idx // (s * s)

        grid_size_x = im_shape[2] / s
        grid_size_y = im_shape[1] / s
        grid_size_z = im_shape[0] / s

        x = (enc_bb[grid_idx, 1] + grid_x) * grid_size_x
        y = (enc_bb[grid_idx, 2] + grid_y) * grid_size_y
        z = (enc_bb[grid_idx, 3] + grid_z) * grid_size_z

        w = enc_bb[grid_idx, 4] * grid_size_x
        h = enc_bb[grid_idx, 5] * grid_size_y
        t = enc_bb[grid_idx, 6] * grid_size_z

        label = np.argmax(enc_class[grid_idx, :])

        pos_bb = [label, (x, y, z, w, h, t)]
        pos_bbs.append(pos_bb)

    return pos_bbs


def crop_image_and_roi(img_vol, all_roi_data, window_shape=[32, 256, 256], shift_aug=False):
    # random roi
    pos_idx = np.random.randint(len(all_roi_data))
    pos_roi = all_roi_data[pos_idx]

    # pull out xyzwht and label
    pos_roi_cw = convert_roidata_roicw(pos_roi)
    pos_roi_label = pos_roi[0]

    # crop image
    [x1, x2, y1, y2, z1, z2] = sample_pos_window(pos_roi_cw, img_vol.shape, window_shape)
    if shift_aug == True:
        pos_img = np.float32(crop_image_with_shifts(img_vol, [x1, x2, y1, y2, z1, z2]))
    else:
        pos_img = np.float32(img_vol[z1:z2, y1:y2, x1:x2])

    # convert bounding box to coords in cropped image
    pos_roi_cw_crop = np.asarray(pos_roi_cw)
    pos_roi_cw_crop[0:3] -= np.asarray([x1, y1, z1])
    pos_bbs = [[pos_roi_label, pos_roi_cw_crop], ]

    # check if more MVBs in image
    other_rois = all_roi_data[:pos_idx] + all_roi_data[pos_idx + 1:]
    other_bbs = add_bbs_to_window([x1, x2, y1, y2, z1, z2], pos_roi, other_rois)

    # convert bounding box to coords in cropped image
    for other_bb_idx in range(len(other_bbs)):
        other_bbs[other_bb_idx][1][0:3] -= np.asarray([x1, y1, z1])
    pos_bbs += other_bbs

    return pos_img, pos_bbs


def crop_image_with_shifts(img_vol, crop_coords, sigma=5, fraction=0.2):
    [x1, x2, y1, y2, z1, z2] = crop_coords
    window_shape = [z2 - z1, y2 - y1, x2 - x1]
    im_shape = img_vol.shape

    img_slices = []
    for z_pos in range(z1, z2):

        if np.random.random() < fraction:
            x_shift = int(np.random.normal(0, sigma))
            y_shift = int(np.random.normal(0, sigma))

            x1_shift = x1 + x_shift
            x2_shift = x2 + x_shift
            y1_shift = y1 + y_shift
            y2_shift = y2 + y_shift

            # take care of out-of-bounds
            if x1_shift < 0:
                x1_shift = 0
                x2_shift = window_shape[2]
            elif x2_shift > im_shape[2]:
                x2_shift = im_shape[2]
                x1_shift = im_shape[2] - window_shape[2]

            if y1_shift < 0:
                y1_shift = 0
                y2_shift = window_shape[1]
            elif y2_shift > im_shape[1]:
                y2_shift = im_shape[1]
                y1_shift = im_shape[1] - window_shape[1]

            img_slice = np.squeeze(img_vol[z_pos, y1_shift:y2_shift, x1_shift:x2_shift])

        else:
            img_slice = np.squeeze(img_vol[z_pos, y1:y2, x1:x2])

        img_slices.append(img_slice)

    return np.stack(img_slices, axis=0)


# def sample_pos_batch(img_vol, all_roi_data, batch_size, s, num_classes, augment=True):
#     pos_imgs = []
#     pos_labels = []
#
#     for i in range(batch_size):
#
#         pos_img, pos_bbs = crop_image_and_roi(img_vol, all_roi_data)
#
#         # augment image and bbs before encoding bbs and appending to list
#         if augment == True:
#             # augments bbs in place, returns new view into image
#             pos_img = augment_image(pos_img, pos_bbs)
#
#         # encode bbs into target label
#         pos_label = encode_pos_bbs(pos_bbs, pos_img.shape, s, num_classes)
#
#         # append
#         pos_imgs.append(pos_img)
#         pos_labels.append(pos_label)
#
#     imgs = np.stack(pos_imgs)
#     imgs = imgs[:, :, :, :, np.newaxis]
#
#     labels = np.stack(pos_labels)
#
#     return imgs, labels


def sample_pos_batch_noncontig(img_vol_list, all_roi_data_list, batch_size, s, num_classes, augment=True):
    # determine prob of selecting from each item in the list of img_vol and all_roi_data
    num_rois = []
    for roi_sublist in all_roi_data_list:
        num_rois.append(len(roi_sublist))
    num_rois = np.array(num_rois)
    prob_list = np.float32(num_rois) / np.sum(num_rois)

    pos_imgs = []
    pos_labels = []

    for i in range(batch_size):

        # randomly select volume
        list_idx = np.random.choice(prob_list.size, p=prob_list)
        img_vol = img_vol_list[list_idx]
        all_roi_data = all_roi_data_list[list_idx]

        # sample positive image, and augment if flag set
        if augment == True:
            # sample with shifted slices
            pos_img, pos_bbs = crop_image_and_roi(img_vol, all_roi_data, shift_aug=True)
            # randomly degrade some slices
            pos_img = degrade_slices_augmentation(pos_img)
            # flip image and bbs.  note: augments bbs in place, returns new view into image
            pos_img = flip_augmentation(pos_img, pos_bbs)
        else:
            pos_img, pos_bbs = crop_image_and_roi(img_vol, all_roi_data)

        # encode bbs into target label
        pos_label = encode_pos_bbs(pos_bbs, pos_img.shape, s, num_classes)

        # append
        pos_imgs.append(pos_img)
        pos_labels.append(pos_label)

    imgs = np.stack(pos_imgs)
    imgs = imgs[:, :, :, :, np.newaxis]

    labels = np.stack(pos_labels)

    return imgs, labels


def degrade_slices_augmentation(img_vol, degrade_fraction=0.2):

    img_slices = []
    for slice_num in range(img_vol.shape[0]):

        img_slice = np.squeeze(img_vol[slice_num, :, :])

        if np.random.random() < degrade_fraction:
            if np.random.random() < 0.33:
                # add noise to slice
                img_slice = img_slice + np.random.normal(0., 10., size=img_slice.shape)
            elif np.random.random() < 0.66:
                # gaussian blur slice
                img_slice = skimage.filters.gaussian(img_slice, sigma=1.5, preserve_range=True)
            else:
                # blackout slice
                img_slice = 0. * img_slice

        img_slices.append(img_slice)

    return np.stack(img_slices, axis=0)


# def sample_neg_batch(img_vol, all_roi_data, batch_size, s, num_classes, augment=True):
#     neg_imgs = []
#     neg_labels = []
#
#     for i in range(batch_size):
#
#         # crop image
#         [x1, x2, y1, y2, z1, z2] = sample_neg_window(img_vol.shape, all_roi_data)
#         neg_img = np.float32(img_vol[z1:z2, y1:y2, x1:x2])
#         neg_bbs = []
#
#         # augment image and bbs before encoding bbs and appending to list
#         if augment == True:
#             neg_img = augment_image(neg_img, neg_bbs)
#
#         # encode negative bbs into target label
#         neg_label = encode_pos_bbs(neg_bbs, neg_img.shape, s, num_classes)
#
#         neg_imgs.append(neg_img)
#         neg_labels.append(neg_label)
#
#     imgs = np.stack(neg_imgs)
#     imgs = imgs[:, :, :, :, np.newaxis]
#
#     labels = np.stack(neg_labels)
#
#     return imgs, labels


def sample_neg_batch_noncontig(img_vol_list, all_roi_data_list, batch_size, s, num_classes, augment=True):
    # determine prob of selecting from each item in the list of img_vol and all_roi_data
    num_rois = []
    for roi_sublist in all_roi_data_list:
        num_rois.append(len(roi_sublist))
    num_rois = np.array(num_rois)
    prob_list = np.float32(num_rois) / np.sum(num_rois)

    neg_imgs = []
    neg_labels = []

    for i in range(batch_size):

        # randomly select volume
        list_idx = np.random.choice(prob_list.size, p=prob_list)
        img_vol = img_vol_list[list_idx]
        all_roi_data = all_roi_data_list[list_idx]

        # crop image
        [x1, x2, y1, y2, z1, z2] = sample_neg_window(img_vol.shape, all_roi_data)
        neg_img = np.float32(img_vol[z1:z2, y1:y2, x1:x2])
        neg_bbs = []

        # augment image and bbs before encoding bbs and appending to list
        if augment == True:
            # randomly degrade some slices
            neg_img = degrade_slices_augmentation(neg_img)
            # flip image and bbs.  note: augments bbs in place, returns new view into image
            neg_img = flip_augmentation(neg_img, neg_bbs)

        # encode negative bbs into target label
        neg_label = encode_pos_bbs(neg_bbs, neg_img.shape, s, num_classes)

        neg_imgs.append(neg_img)
        neg_labels.append(neg_label)

    imgs = np.stack(neg_imgs)
    imgs = imgs[:, :, :, :, np.newaxis]

    labels = np.stack(neg_labels)

    return imgs, labels


def flip_augmentation(img, bbs):
    # rotate
    if random.random() < 0.5:
        # print('transpose')
        img = np.transpose(img, (0, 2, 1))
        for idx, bb in enumerate(bbs):
            bb_coords = bb[1]
            xp = bb_coords[1]
            yp = bb_coords[0]
            zp = bb_coords[2]
            wp = bb_coords[4]
            hp = bb_coords[3]
            tp = bb_coords[5]
            bbs[idx][1] = [xp, yp, zp, wp, hp, tp]

    # flip X
    if random.random() < 0.5:
        # print('flip X')
        img = img[:, :, ::-1]
        for idx, bb in enumerate(bbs):
            bb_coords = bb[1]
            xp = (img.shape[2] - 1) - bb_coords[0]
            yp = bb_coords[1]
            zp = bb_coords[2]
            wp = bb_coords[3]
            hp = bb_coords[4]
            tp = bb_coords[5]
            bbs[idx][1] = [xp, yp, zp, wp, hp, tp]

    # flip Y
    if random.random() < 0.5:
        # print('flip Y')
        img = img[:, ::-1, :]
        for idx, bb in enumerate(bbs):
            bb_coords = bb[1]
            xp = bb_coords[0]
            yp = (img.shape[1] - 1) - bb_coords[1]
            zp = bb_coords[2]
            wp = bb_coords[3]
            hp = bb_coords[4]
            tp = bb_coords[5]
            bbs[idx][1] = [xp, yp, zp, wp, hp, tp]

    # flip Z
    if random.random() < 0.5:
        # print('flip Z')
        img = img[::-1, :, :]
        for idx, bb in enumerate(bbs):
            bb_coords = bb[1]
            xp = bb_coords[0]
            yp = bb_coords[1]
            zp = (img.shape[0] - 1) - bb_coords[2]
            wp = bb_coords[3]
            hp = bb_coords[4]
            tp = bb_coords[5]
            bbs[idx][1] = [xp, yp, zp, wp, hp, tp]

    return img


def decode_prediction(yolo_output, im_shape, s, num_classes, conf_thresh):
    num_bbs = 2

    pred_classes = np.reshape(yolo_output[:s * s * s * num_classes], (s * s * s, num_classes))
    pred_bb = np.reshape(yolo_output[s * s * s * num_classes:], (s * s * s, num_bbs * 7))

    pos_bbs = []
    confs = []
    for grid_idx in range(s * s * s):

        max_conf = np.maximum(pred_bb[grid_idx, 0], pred_bb[grid_idx, 7])
        argmax_conf = np.argmax(np.array(pred_bb[grid_idx, 0], pred_bb[grid_idx, 7]))

        if max_conf < conf_thresh:
            continue

        grid_x = (grid_idx % (s * s)) % s
        grid_y = (grid_idx % (s * s)) // s
        grid_z = grid_idx // (s * s)

        grid_size_x = im_shape[2] / s
        grid_size_y = im_shape[1] / s
        grid_size_z = im_shape[0] / s

        x = (pred_bb[grid_idx, argmax_conf * 7 + 1] + grid_x) * grid_size_x
        y = (pred_bb[grid_idx, argmax_conf * 7 + 2] + grid_y) * grid_size_y
        z = (pred_bb[grid_idx, argmax_conf * 7 + 3] + grid_z) * grid_size_z

        w = pred_bb[grid_idx, argmax_conf * 7 + 4] * grid_size_x
        h = pred_bb[grid_idx, argmax_conf * 7 + 5] * grid_size_y
        t = pred_bb[grid_idx, argmax_conf * 7 + 6] * grid_size_z

        label = np.argmax(pred_classes[grid_idx, :])

        pos_bb = [label, [x, y, z, w, h, t]]
        pos_bbs.append(pos_bb)
        confs.append(max_conf)

    return pos_bbs, confs


def drawBB(img, boxes):
    PIL_img = Image.fromarray(img)
    draw = ImageDraw.Draw(PIL_img)

    x0 = boxes[0] - 0.5 * boxes[2]
    x1 = boxes[0] + 0.5 * boxes[2]
    y0 = boxes[1] - 0.5 * boxes[3]
    y1 = boxes[1] + 0.5 * boxes[3]

    xy = [(x0, y0), (x1, y1)]
    draw.rectangle(xy, outline='white')

    return np.array(PIL_img)


# show_pred is untested since refactoring
# def show_pred(img_vol, pred_bbs):
#     img_slices = []
#
#     for i in range(len(pred_bbs)):
#         slice_num = int(pred_bbs[i][1][2])
#         box_2d = np.asarray([int(pred_bbs[i][1][0]),
#                              int(pred_bbs[i][1][1]),
#                              int(pred_bbs[i][1][3]),
#                              int(pred_bbs[i][1][4])])
#
#         img_slice = np.squeeze(img_vol[slice_num, :, :])
#         img_slice = drawBB(img_slice, box_2d)
#
#         img_slices.append(img_slice)
#
#     return img_slices


def inference_on_large_volume(sess, pred, tf_data, keep_prob, img_vol, conf_thresh, batch_size, crop_size, s, num_classes):

    dx = crop_size[2] // 2
    dy = crop_size[1] // 2
    dz = crop_size[0] // 2
    all_x = np.arange(0, img_vol.shape[2] - crop_size[2] + 1, dx)
    all_y = np.arange(0, img_vol.shape[1] - crop_size[1] + 1, dy)
    all_z = np.arange(0, img_vol.shape[0] - crop_size[0] + 1, dz)

    # check for edges
    if img_vol.shape[2] % dx != 0:
        all_x = np.append(all_x, img_vol.shape[2] - crop_size[2])
    if img_vol.shape[1] % dy != 0:
        all_y = np.append(all_y, img_vol.shape[1] - crop_size[1])
    if img_vol.shape[0] % dz != 0:
        all_z = np.append(all_z, img_vol.shape[0] - crop_size[0])

    # collect all coordinate to visit in img_vol
    all_coords = []
    for x in all_x:
        for y in all_y:
            for z in all_z:
                all_coords.append([x, y, z])

    # randomly shuffle positions, for mixed batches
    random.shuffle(all_coords)

    # split positions into batches
    num_batches = len(all_coords) // batch_size
    print("num_batches = " + str(num_batches))

    all_pred_BBs = []
    all_confs = []

    for batch_num in range(num_batches):

        batch_imgs = []

        for i in range(batch_size):
            coord_idx = batch_num * batch_size + i

            # crop image
            x1 = all_coords[coord_idx][0]
            y1 = all_coords[coord_idx][1]
            z1 = all_coords[coord_idx][2]
            # note: image volumes are (z,y,x) because of tensorflow
            x2 = x1 + crop_size[2]
            y2 = y1 + crop_size[1]
            z2 = z1 + crop_size[0]
            img_crop = np.float32(img_vol[z1:z2, y1:y2, x1:x2])

            batch_imgs.append(img_crop)

        batch_imgs = np.stack(batch_imgs)
        batch_imgs = batch_imgs[:, :, :, :, np.newaxis]

        # preprocess
        batch_imgs = preprocess_image(batch_imgs)

        # inference
        output = sess.run(pred, feed_dict={tf_data: batch_imgs, keep_prob: 1.})

        # parse output
        for i in range(batch_size):

            # convert prediction to: (x,y,z,w,h,t)
            pred_BBs, confs = decode_prediction(output[i, :], crop_size, s, num_classes, conf_thresh)

            coord_idx = batch_num * batch_size + i

            for pred_num in range(len(pred_BBs)):
                pred_BB_label = pred_BBs[pred_num][0]
                pred_BB_coords = np.array(pred_BBs[pred_num][1])
                pred_BB_coords[0:3] += np.array(all_coords[coord_idx])
                all_pred_BBs.append([pred_BB_label, pred_BB_coords])
                all_confs.append(confs[pred_num])

    # run through final (partial) batch
    rem_coords = len(all_coords) % batch_size
    if rem_coords:

        batch_imgs = []

        for i in range(rem_coords):
            coord_idx = num_batches * batch_size + i

            # crop image
            x1 = all_coords[coord_idx][0]
            y1 = all_coords[coord_idx][1]
            z1 = all_coords[coord_idx][2]
            # note: image volumes are (z,y,x) because of tensorflow
            x2 = x1 + crop_size[2]
            y2 = y1 + crop_size[1]
            z2 = z1 + crop_size[0]
            img_crop = np.float32(img_vol[z1:z2, y1:y2, x1:x2])

            batch_imgs.append(img_crop)

        # fill rest of batch with random images
        for i in range(batch_size - rem_coords):
            # randomly sample other images
            coord_idx = np.random.random_integers(len(all_coords))

            # crop image
            x1 = all_coords[coord_idx][0]
            y1 = all_coords[coord_idx][1]
            z1 = all_coords[coord_idx][2]
            # note: image volumes are (z,y,x) because of tensorflow
            x2 = x1 + crop_size[2]
            y2 = y1 + crop_size[1]
            z2 = z1 + crop_size[0]
            img_crop = np.float32(img_vol[z1:z2, y1:y2, x1:x2])

            batch_imgs.append(img_crop)

        batch_imgs = np.stack(batch_imgs)
        batch_imgs = batch_imgs[:, :, :, :, np.newaxis]

        # preprocess
        batch_imgs = preprocess_image(batch_imgs)

        # inference
        output = sess.run(pred, feed_dict={tf_data: batch_imgs, keep_prob: 1.})

        # only parse output from the few new images
        for i in range(rem_coords):

            # convert prediction to: (x,y,z,w,h,t)
            pred_BBs, confs = decode_prediction(output[i, :], crop_size, s, num_classes, conf_thresh)

            coord_idx = num_batches * batch_size + i

            for pred_num in range(len(pred_BBs)):
                pred_BB_label = pred_BBs[pred_num][0]
                pred_BB_coords = np.array(pred_BBs[pred_num][1])
                pred_BB_coords[0:3] += np.array(all_coords[coord_idx])
                all_pred_BBs.append([pred_BB_label, pred_BB_coords])
                all_confs.append(confs[pred_num])

    return all_pred_BBs, all_confs


def NMS(bbs, confs, iou_thresh=0.5):
    rm_idx = [0] * len(bbs)

    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):

            iou_ij = iou_np(bbs[i][1], bbs[j][1])

            if iou_ij > iou_thresh:

                rm_ij = np.argmin(np.array([confs[i], confs[j]]))
                if rm_ij == 0:
                    rm_idx[i] = 1
                    break
                else:
                    rm_idx[j] = 1

    unique_bbs = [x for x, y in zip(bbs, rm_idx) if not y]
    unique_confs = [x for x, y in zip(confs, rm_idx) if not y]

    return unique_bbs, unique_confs


def NMS_fast(bbs, confs, iou_thresh=0.5):

    # sort BBs by increasing confidence, so that lowest confidence is removed first
    confs, bbs = (list(t) for t in zip(*sorted(zip(confs, bbs), key=lambda pair: pair[0])))

    rm_idx = [0] * len(bbs)

    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):

            # calc iou
            if iou_np_fast(bbs[i][1], bbs[j][1]) > iou_thresh:
                rm_idx[i] = 1
                break

    unique_bbs = [x for x, y in zip(bbs, rm_idx) if not y]
    unique_confs = [x for x, y in zip(confs, rm_idx) if not y]

    return unique_bbs, unique_confs


#### EVALUATION ####

def evaluation(pred_bbs, confs, gt_rois, det_thresh):

    # sort pred_bbs by decreasing confidence, so that highest confidence is assigned first
    confs, pred_bbs = (list(t) for t in zip(*sorted(zip(confs, pred_bbs), key=lambda pair: pair[0], reverse=True)))

    pred_hits = np.zeros(len(pred_bbs))
    gt_hits = np.zeros(len(gt_rois))

    for pred_idx, pred_bb in enumerate(pred_bbs):

        for gt_idx, gt_roi in enumerate(gt_rois):

            # if gt_roi already detected, skip
            if gt_hits[gt_idx]:
                continue

            # convert gt roi to bb
            gt_bb_cw = convert_roidata_roicw(gt_roi)

            # extract coords from pred_bb
            pred_bb_cw = pred_bb[1]

            # check for detection
            if iou_np(pred_bb_cw, gt_bb_cw) > det_thresh:
                gt_hits[gt_idx] = 1
                pred_hits[pred_idx] = 1
                break

    #
    TP = pred_hits.sum()
    FP = pred_hits.size - pred_hits.sum()

    return TP, FP


def evaluation_scan_conf(pred_bbs, confs, gt_rois, conf_thresh_list, det_thresh):
    TP_list, FP_list = [], []

    for conf_thresh in conf_thresh_list:
        filt_pred_bbs = [x for x, y in zip(pred_bbs, confs) if y > conf_thresh]
        filt_confs = [y for y in confs if y > conf_thresh]

        TP, FP = evaluation(filt_pred_bbs, filt_confs, gt_rois, det_thresh)

        TP_list.append(TP)
        FP_list.append(FP)

    return TP_list, FP_list


def filt_bbs(pred_bbs, confs, conf_thresh):

    filt_pred_bbs = [x for x, y in zip(pred_bbs, confs) if y > conf_thresh]
    filt_confs = [y for y in confs if y > conf_thresh]

    return filt_pred_bbs, filt_confs


def subsample_data_and_rois(img_vol, train_roi_data, val_roi_data, img_coords_lr, roi_coords_lr):

    x1 = roi_coords_lr[0]
    x2 = roi_coords_lr[1]
    y1 = roi_coords_lr[2]
    y2 = roi_coords_lr[3]
    z1 = roi_coords_lr[4]
    z2 = roi_coords_lr[5]

    # crop image
    img_vol_crop = img_vol[img_coords_lr[4]:img_coords_lr[5],
                   img_coords_lr[2]:img_coords_lr[3],
                   img_coords_lr[0]:img_coords_lr[1]]

    train_roi_data_crop = []
    for roi in train_roi_data:
        roi_cw = convert_roidata_roicw(roi)
        if (roi_cw[0] >= x1) and (roi_cw[0] < x2):
            if (roi_cw[1] >= y1) and (roi_cw[1] < y2):
                if (roi_cw[2] >= z1) and (roi_cw[2] < z2):
                    # account for shift from cropping image
                    # unfortunately, rois are stored as tuples.  need to convert to lists and back
                    tmp_roi = [roi[0], list(roi[1])]
                    tmp_roi[1][1] -= img_coords_lr[0]
                    tmp_roi[1][2] -= img_coords_lr[2]
                    tmp_roi[1][0] -= img_coords_lr[4]
                    train_roi_data_crop.append(tuple([tmp_roi[0], tuple(tmp_roi[1])]))

    val_roi_data_crop = []
    for roi in val_roi_data:
        roi_cw = convert_roidata_roicw(roi)
        if (roi_cw[0] >= x1) and (roi_cw[0] < x2):
            if (roi_cw[1] >= y1) and (roi_cw[1] < y2):
                if (roi_cw[2] >= z1) and (roi_cw[2] < z2):
                    # account for shift from cropping image
                    # unfortunately, rois are stored as tuples.  need to convert to lists and back
                    tmp_roi = [roi[0], list(roi[1])]
                    tmp_roi[1][1] -= img_coords_lr[0]
                    tmp_roi[1][2] -= img_coords_lr[2]
                    tmp_roi[1][0] -= img_coords_lr[4]
                    val_roi_data_crop.append(tuple([tmp_roi[0], tuple(tmp_roi[1])]))

    return img_vol_crop, train_roi_data_crop, val_roi_data_crop


def save_TP_FP_FN(outpath, img_vol, pred_bbs, confs, gt_rois, det_thresh, crop_size=[32, 256, 256]):
    # first create list of TP, FP, FN ------------------------------------------------------------------

    # sort pred_bbs by decreasing confidence, so that highest confidence is assigned first
    confs, pred_bbs = (list(t) for t in zip(*sorted(zip(confs, pred_bbs), key=lambda pair: pair[0], reverse=True)))

    # keep track of which train/val examples have already been detected
    pred_hits = np.zeros(len(pred_bbs))
    gt_hits = np.zeros(len(gt_rois))

    # store BBs
    TP_list = []
    FN_list = []
    FP_list = []

    # TPs and FPs
    for pred_idx, pred_bb in enumerate(pred_bbs):

        for gt_idx, gt_roi in enumerate(gt_rois):

            # if train_roi already detected, skip
            if gt_hits[gt_idx]:
                continue

            # convert train roi to bb
            gt_bb_cw = convert_roidata_roicw(gt_roi)

            # extract coords from pred_bb
            pred_bb_cw = pred_bb[1]

            # check for detection
            if iou_np(pred_bb_cw, gt_bb_cw) > det_thresh:
                TP_list.append(pred_bb)
                gt_hits[gt_idx] = 1
                pred_hits[pred_idx] = 1
                break

        # if not assigned to any gt_roi, then FP
        if pred_hits[pred_idx] == 0:
            FP_list.append(pred_bb)

    # FNs
    for gt_idx, gt_roi in enumerate(gt_rois):
        if gt_hits[gt_idx] == 0:
            gt_bb_cw = convert_roidata_roicw(gt_roi)
            FN_list.append(gt_bb_cw)

    num_TP = pred_hits.sum()
    num_FP = pred_hits.size - pred_hits.sum()
    num_FN = gt_hits.size - gt_hits.sum()

    print('number of true positives = ' + str(num_TP))
    print('number of false positives = ' + str(num_FP))
    print('number of false negatives = ' + str(num_FN))

    # create and save images ---------------------------------------------------------

    TP_path = os.path.join(outpath, 'TPs')
    FN_path = os.path.join(outpath, 'FNs')
    FP_path = os.path.join(outpath, 'FPs')

    all_lists = [TP_list, FN_list, FP_list]
    list_names = ['TP', 'FN', 'FP']
    list_paths = [TP_path, FN_path, FP_path]

    for list_idx, list_ in enumerate(all_lists):

        os.mkdir(list_paths[list_idx])

        count = 0

        for bb in list_:

            # crop image
            [x1, x2, y1, y2, z1, z2] = sample_pos_window(bb, img_vol.shape, crop_size)
            crop_img = img_vol[z1:z2, y1:y2, x1:x2]

            # shift bb relative to crop
            bb_cw = bb[1]
            bb_cw[0:3] -= np.asarray([x1, y1, z1])

            # draw box into crop_img
            img_slices = []
            box_2d = np.asarray([int(bb_cw[0]), int(bb_cw[1]), int(bb_cw[3]), int(bb_cw[4])])

            for slice_num in range(crop_img.shape[0]):

                img_slice = np.squeeze(crop_img[slice_num, :, :])
                bb_lr = convert_cw_lr(bb_cw)
                if (slice_num >= int(bb_lr[4])) and (slice_num < int(bb_lr[5])):
                    img_slice = drawBB(img_slice, box_2d)
                img_slices.append(img_slice)

            crop_img = np.stack(img_slices)

            fname = os.path.join(list_paths[list_idx], list_names[list_idx] + '_' + str(count) + '.tif')
            skimage.io.imsave(fname, crop_img)
            count += 1

    return 0


def random_batch(img_vol, batch_size, crop_size):

    batch_imgs = []
    coords = []

    for i in range(batch_size):

        # crop image
        # note: image volumes are (z,y,x) because of tensorflow
        x1 = np.random.randint(0, img_vol.shape[2]-crop_size[2])
        y1 = np.random.randint(0, img_vol.shape[1]-crop_size[1])
        z1 = np.random.randint(0, img_vol.shape[0]-crop_size[0])
        x2 = x1 + crop_size[2]
        y2 = y1 + crop_size[1]
        z2 = z1 + crop_size[0]
        img_crop = np.float32(img_vol[z1:z2, y1:y2, x1:x2])

        batch_imgs.append(img_crop)
        coords.append([x1, y1, z1])

    batch_imgs = np.stack(batch_imgs)
    batch_imgs = batch_imgs[:, :, :, :, np.newaxis]

    # preprocess
    batch_imgs = preprocess_image(batch_imgs)

    return batch_imgs, coords
