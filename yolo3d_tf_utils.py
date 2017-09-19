import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d(x, w):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')
    

def pool2d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')


def pool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def iou_tf(box1, box2):
    
    # intersection --------------------------------------------------------------------------------
    
    x_int = tf.minimum(box1[1]+0.5*box1[4], box2[1]+0.5*box2[4]) - tf.maximum(box1[1]-0.5*box1[4], box2[1]-0.5*box2[4])
    y_int = tf.minimum(box1[2]+0.5*box1[5], box2[2]+0.5*box2[5]) - tf.maximum(box1[2]-0.5*box1[5], box2[2]-0.5*box2[5])
    z_int = tf.minimum(box1[3]+0.5*box1[6], box2[3]+0.5*box2[6]) - tf.maximum(box1[3]-0.5*box1[6], box2[3]-0.5*box2[6])

    intersection = tf.multiply(x_int, y_int)
    intersection = tf.multiply(intersection, z_int)
    
    # account for negative values of x_int, y_int, z_int
    neg_x_int = tf.less(x_int, 0)
    neg_y_int = tf.less(y_int, 0)
    neg_z_int = tf.less(z_int, 0)
    
    # account for negative values of w,h,t (in prediction: 2nd arg)
    neg_w = tf.less(box2[4], 0)
    neg_h = tf.less(box2[5], 0)
    neg_t = tf.less(box2[6], 0)
    
    # remove all possible problems with intersection arising from neg values
    neg_xy = tf.logical_or(neg_x_int, neg_y_int)
    neg_xyz = tf.logical_or(neg_xy, neg_z_int)
    
    neg_wh = tf.logical_or(neg_w, neg_h)
    neg_wht = tf.logical_or(neg_wh, neg_t)
    neg = tf.logical_or(neg_xyz, neg_wht)
    
    pos = tf.cast(tf.logical_not(neg), tf.float32)
    intersection = tf.multiply(intersection, pos)
    
    # union -----------------------------------------------------------------------------------------
    
    # volumes
    vol1 = tf.multiply(box1[4], box1[5])
    vol1 = tf.multiply(vol1, box1[6])

    vol2 = tf.multiply(box2[4], box2[5])
    vol2 = tf.multiply(vol2, box2[6])
    
    # account for no box in vol1
    zero_vol1 = tf.equal(vol1, tf.zeros_like(vol1))
    vol1 = vol1 + tf.cast(zero_vol1, tf.float32)

    # account for possible negative volume in box2
    vol2 = tf.abs(vol2)
    
    # account for tiny box in vol2 ( < eps )
    eps = 1e-7
    zero_vol2 = tf.less_equal(vol2, eps*tf.ones_like(vol2))
    vol2 = vol2 + tf.cast(zero_vol2, tf.float32)

    union = vol1 + vol2 - intersection

    return tf.divide(intersection, union)
    

def yolo_loss(pred, labels, s, num_bb, num_classes, lambda_weights=(10., 0.5, 0.1, 50.)):

    # this loss has only been tested for num_bb = 2.  This is the architecture used in the paper.
    # no guarantees that the code will run if you change num_bb.
    assert(num_bb == 2)

    # parse the weights
    lambda_conf = lambda_weights[0]
    lambda_noobj = lambda_weights[1]
    lambda_class = lambda_weights[2]
    lambda_coord = lambda_weights[3]

    # separate classes from BBs and reshape -------------------------------------------------------------------

    class_labels = tf.slice(labels, [0, 0], [-1, s * s * s * num_classes])
    BB_labels = tf.slice(labels, [0, s * s * s * num_classes], [-1, -1])
    BB_labels = tf.reshape(BB_labels, [-1, s * s * s, 7])

    class_pred = tf.slice(pred, [0, 0], [-1, s * s * s * num_classes])
    BB_pred = tf.slice(pred, [0, s * s * s * num_classes], [-1, -1])
    BB_pred = tf.reshape(BB_pred, [-1, s * s * s, num_bb * 7])

    # figure out which grid element has an object in it ----------------------------------------------------------

    obj_vec = tf.squeeze(tf.slice(BB_labels, [0, 0, 0], [-1, -1, 1]))
    noobj_vec = tf.squeeze(tf.ones_like(obj_vec) - obj_vec)

    # tile to matrix
    obj_mat = tf.tile(tf.reshape(obj_vec, [-1, s * s * s, 1]), [1, 1, num_bb * 7])
    noobj_mat = tf.tile(tf.reshape(noobj_vec, [-1, s * s * s, 1]), [1, 1, num_bb * 7])

    # IOU of each BB to determine which is "responsible" -------------------------------------------------------

    # unpack BBs from BB_pred and BB_labels
    box_labels = tf.unstack(BB_labels, axis=2)

    box_pred1 = tf.slice(BB_pred, [0, 0, 0], [-1, -1, 7])
    box_pred1 = tf.unstack(box_pred1, axis=2)

    box_pred2 = tf.slice(BB_pred, [0, 0, 7], [-1, -1, 7])
    box_pred2 = tf.unstack(box_pred2, axis=2)

    # calculate IOU from both BBs
    iou_pred1 = iou_tf(box_labels, box_pred1)
    iou_pred2 = iou_tf(box_labels, box_pred2)

    # find argmax for every BB
    iou_preds = tf.pack([iou_pred1, iou_pred2], axis=2)
    bb_idx = tf.argmax(iou_preds, axis=2)

    # construct target BB and append target classes -------------------------------------------------------

    # select the higher confidence and construct the target_bb
    conf = tf.maximum(iou_pred1, iou_pred2)
    target_bb = tf.pack([conf, box_labels[1], box_labels[2], box_labels[3], box_labels[4], box_labels[5],
                         box_labels[6]], axis=2)
    target_bb = tf.tile(target_bb, [1, 1, num_bb])
    target_bb = tf.multiply(target_bb, obj_mat)
    target_bb = tf.reshape(target_bb, [-1, s * s * s * num_bb * 7])

    # target classes
    target_class = class_labels
    target = tf.concat(1, [target_class, target_bb])

    # Weight matrix to silence loss on elements without object and BBs that aren't responsible ------------

    # weights for loss
    obj_weights_vec = tf.reshape(tf.constant([lambda_conf, lambda_coord, lambda_coord, lambda_coord, lambda_coord,
                                              lambda_coord, lambda_coord]), [1, 7])
    obj_weights_mat = tf.tile(tf.tile(obj_weights_vec, [s * s * s, 1]), [1, num_bb])

    noobj_weights_vec = tf.reshape(tf.constant([lambda_noobj, 0., 0., 0., 0., 0., 0.]), [1, 7])
    noobj_weights_mat = tf.tile(tf.tile(noobj_weights_vec, [s * s * s, 1]), [1, num_bb])

    # figure out which BB is responsible
    obj_bb1_vec = tf.multiply(obj_vec, tf.to_float(tf.ones_like(bb_idx)-bb_idx))
    obj_bb1_mat = tf.tile(tf.reshape(obj_bb1_vec, [-1, s * s * s, 1]), [1, 1, 7])

    obj_bb2_vec = tf.multiply(obj_vec, tf.to_float(bb_idx))
    obj_bb2_mat = tf.tile(tf.reshape(obj_bb2_vec, [-1, s * s * s, 1]), [1, 1, 7])

    obj_bb12_mat = tf.concat(2, [obj_bb1_mat, obj_bb2_mat])

    # combine objects, responsible BB, and weights
    weights_bb = tf.multiply(noobj_weights_mat, noobj_mat) + tf.multiply(obj_weights_mat, obj_bb12_mat)
    weights_bb = tf.reshape(weights_bb, [-1, s * s * s * num_bb * 7])

    # take care of class weights
    weights_class_mat = tf.tile(tf.reshape(obj_vec, [-1, s * s * s, 1]), [1, 1, num_classes])
    weights_class_mat = weights_class_mat * lambda_class
    weights_class = tf.reshape(weights_class_mat, [-1, s * s * s * num_classes])

    # combine all weights
    weights = tf.concat(1, [weights_class, weights_bb])

    loss = tf.reduce_mean(tf.multiply(weights, tf.square(pred-target)))

    return loss
