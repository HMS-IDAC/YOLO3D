import tensorflow as tf
from yolo3d_tf_utils import weight_variable, bias_variable, conv3d, pool3d, pool2d


def yolo_L1_16(data, s, num_bb, num_classes, keep_prob):

    # note: this network has s = 4, based on input size (32,256,256) and number of max pools
    assert(s == 4)

    # slope of leaky ReLU
    alpha = tf.constant(0.1)

    # conv
    W_conv0 = weight_variable([5, 5, 5, 1, 16])
    z_conv0 = conv3d(data, W_conv0)
    batch_mean, batch_var = tf.nn.moments(z_conv0, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv0_bn = tf.nn.batch_normalization(z_conv0, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv0 = tf.maximum(z_conv0_bn, tf.multiply(alpha, z_conv0_bn))

    # pool
    h_pool0 = pool2d(h_conv0)

    # conv
    W_conv1 = weight_variable([3, 3, 3, 16, 16])
    z_conv1 = conv3d(h_pool0, W_conv1)
    batch_mean, batch_var = tf.nn.moments(z_conv1, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv1_bn = tf.nn.batch_normalization(z_conv1, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv1 = tf.maximum(z_conv1_bn, tf.multiply(alpha, z_conv1_bn))

    # pool
    h_pool1 = pool2d(h_conv1)

    # conv
    W_conv2 = weight_variable([3, 3, 3, 16, 16])
    z_conv2 = conv3d(h_pool1, W_conv2)
    batch_mean, batch_var = tf.nn.moments(z_conv2, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv2_bn = tf.nn.batch_normalization(z_conv2, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv2 = tf.maximum(z_conv2_bn, tf.multiply(alpha, z_conv2_bn))

    # pool
    h_pool2 = pool2d(h_conv2)

    # conv
    W_conv3 = weight_variable([3, 3, 3, 16, 16])
    z_conv3 = conv3d(h_pool2, W_conv3)
    batch_mean, batch_var = tf.nn.moments(z_conv3, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv3_bn = tf.nn.batch_normalization(z_conv3, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv3 = tf.maximum(z_conv3_bn, tf.multiply(alpha, z_conv3_bn))

    # pool
    h_pool3 = pool3d(h_conv3)

    # conv
    W_conv4 = weight_variable([3, 3, 3, 16, 32])
    z_conv4 = conv3d(h_pool3, W_conv4)
    batch_mean, batch_var = tf.nn.moments(z_conv4, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv4_bn = tf.nn.batch_normalization(z_conv4, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv4 = tf.maximum(z_conv4_bn, tf.multiply(alpha, z_conv4_bn))

    # pool
    h_pool4 = pool3d(h_conv4)

    # conv
    W_conv5 = weight_variable([3, 3, 3, 32, 64])
    z_conv5 = conv3d(h_pool4, W_conv5)
    batch_mean, batch_var = tf.nn.moments(z_conv5, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(64))
    gamma = tf.Variable(tf.ones(64))
    z_conv5_bn = tf.nn.batch_normalization(z_conv5, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv5 = tf.maximum(z_conv5_bn, tf.multiply(alpha, z_conv5_bn))

    # pool
    h_pool5 = pool3d(h_conv5)

    # conv
    W_conv6 = weight_variable([3, 3, 3, 64, 128])
    z_conv6 = conv3d(h_pool5, W_conv6)
    batch_mean, batch_var = tf.nn.moments(z_conv6, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(128))
    gamma = tf.Variable(tf.ones(128))
    z_conv6_bn = tf.nn.batch_normalization(z_conv6, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv6 = tf.maximum(z_conv6_bn, tf.multiply(alpha, z_conv6_bn))

    # conv
    W_conv7 = weight_variable([3, 3, 3, 128, 256])
    z_conv7 = conv3d(h_conv6, W_conv7)
    batch_mean, batch_var = tf.nn.moments(z_conv7, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(256))
    gamma = tf.Variable(tf.ones(256))
    z_conv7_bn = tf.nn.batch_normalization(z_conv7, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv7 = tf.maximum(z_conv7_bn, tf.multiply(alpha, z_conv7_bn))

    # conv
    W_conv8 = weight_variable([3, 3, 3, 256, 512])
    z_conv8 = conv3d(h_conv7, W_conv8)
    batch_mean, batch_var = tf.nn.moments(z_conv8, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(512))
    gamma = tf.Variable(tf.ones(512))
    z_conv8_bn = tf.nn.batch_normalization(z_conv8, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv8 = tf.maximum(z_conv8_bn, tf.multiply(alpha, z_conv8_bn))

    # conv (NIN)
    W_conv8N = weight_variable([1, 1, 1, 512, 256])
    z_conv8N = conv3d(h_conv8, W_conv8N)
    batch_mean, batch_var = tf.nn.moments(z_conv8N, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(256))
    gamma = tf.Variable(tf.ones(256))
    z_conv8N_bn = tf.nn.batch_normalization(z_conv8N, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv8N = tf.maximum(z_conv8N_bn, tf.multiply(alpha, z_conv8N_bn))

    # flatten
    h_flat = tf.reshape(h_conv8N, shape=[-1, s * s * s * 256])

    # fc
    W_fc1 = weight_variable([s * s * s * 256, 512])
    z_fc1 = tf.matmul(h_flat, W_fc1)
    batch_mean, batch_var = tf.nn.moments(z_fc1, [0])
    beta = tf.Variable(tf.zeros(512))
    gamma = tf.Variable(tf.ones(512))
    z_fc1_bn = tf.nn.batch_normalization(z_fc1, batch_mean, batch_var, beta, gamma, 1e-5)
    h_fc1 = tf.maximum(z_fc1_bn, tf.multiply(alpha, z_fc1_bn))
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output
    W_score = weight_variable([512, s * s * s * (7 * num_bb + num_classes)])
    b_score = bias_variable([s * s * s * (7 * num_bb + num_classes)])
    pred = tf.add(tf.matmul(h_fc1_drop, W_score), b_score, name='pred')

    return pred


def yolo_L1_12(data, s, num_bb, num_classes, keep_prob):

    # note: this network has s = 4, based on input size (32,256,256) and number of max pools
    assert(s == 4)

    # slope of leaky ReLU
    alpha = tf.constant(0.1)

    # conv
    W_conv0 = weight_variable([5, 5, 5, 1, 12])
    z_conv0 = conv3d(data, W_conv0)
    batch_mean, batch_var = tf.nn.moments(z_conv0, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(12))
    gamma = tf.Variable(tf.ones(12))
    z_conv0_bn = tf.nn.batch_normalization(z_conv0, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv0 = tf.maximum(z_conv0_bn, tf.multiply(alpha, z_conv0_bn))

    # pool
    h_pool0 = pool2d(h_conv0)

    # conv
    W_conv1 = weight_variable([3, 3, 3, 12, 12])
    z_conv1 = conv3d(h_pool0, W_conv1)
    batch_mean, batch_var = tf.nn.moments(z_conv1, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(12))
    gamma = tf.Variable(tf.ones(12))
    z_conv1_bn = tf.nn.batch_normalization(z_conv1, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv1 = tf.maximum(z_conv1_bn, tf.multiply(alpha, z_conv1_bn))

    # pool
    h_pool1 = pool2d(h_conv1)

    # conv
    W_conv2 = weight_variable([3, 3, 3, 12, 16])
    z_conv2 = conv3d(h_pool1, W_conv2)
    batch_mean, batch_var = tf.nn.moments(z_conv2, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv2_bn = tf.nn.batch_normalization(z_conv2, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv2 = tf.maximum(z_conv2_bn, tf.multiply(alpha, z_conv2_bn))

    # pool
    h_pool2 = pool2d(h_conv2)

    # conv
    W_conv3 = weight_variable([3, 3, 3, 16, 16])
    z_conv3 = conv3d(h_pool2, W_conv3)
    batch_mean, batch_var = tf.nn.moments(z_conv3, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv3_bn = tf.nn.batch_normalization(z_conv3, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv3 = tf.maximum(z_conv3_bn, tf.multiply(alpha, z_conv3_bn))

    # pool
    h_pool3 = pool3d(h_conv3)

    # conv
    W_conv4 = weight_variable([3, 3, 3, 16, 32])
    z_conv4 = conv3d(h_pool3, W_conv4)
    batch_mean, batch_var = tf.nn.moments(z_conv4, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv4_bn = tf.nn.batch_normalization(z_conv4, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv4 = tf.maximum(z_conv4_bn, tf.multiply(alpha, z_conv4_bn))

    # pool
    h_pool4 = pool3d(h_conv4)

    # conv
    W_conv5 = weight_variable([3, 3, 3, 32, 64])
    z_conv5 = conv3d(h_pool4, W_conv5)
    batch_mean, batch_var = tf.nn.moments(z_conv5, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(64))
    gamma = tf.Variable(tf.ones(64))
    z_conv5_bn = tf.nn.batch_normalization(z_conv5, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv5 = tf.maximum(z_conv5_bn, tf.multiply(alpha, z_conv5_bn))

    # pool
    h_pool5 = pool3d(h_conv5)

    # conv
    W_conv6 = weight_variable([3, 3, 3, 64, 128])
    z_conv6 = conv3d(h_pool5, W_conv6)
    batch_mean, batch_var = tf.nn.moments(z_conv6, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(128))
    gamma = tf.Variable(tf.ones(128))
    z_conv6_bn = tf.nn.batch_normalization(z_conv6, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv6 = tf.maximum(z_conv6_bn, tf.multiply(alpha, z_conv6_bn))

    # conv
    W_conv7 = weight_variable([3, 3, 3, 128, 256])
    z_conv7 = conv3d(h_conv6, W_conv7)
    batch_mean, batch_var = tf.nn.moments(z_conv7, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(256))
    gamma = tf.Variable(tf.ones(256))
    z_conv7_bn = tf.nn.batch_normalization(z_conv7, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv7 = tf.maximum(z_conv7_bn, tf.multiply(alpha, z_conv7_bn))

    # conv
    W_conv8 = weight_variable([3, 3, 3, 256, 512])
    z_conv8 = conv3d(h_conv7, W_conv8)
    batch_mean, batch_var = tf.nn.moments(z_conv8, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(512))
    gamma = tf.Variable(tf.ones(512))
    z_conv8_bn = tf.nn.batch_normalization(z_conv8, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv8 = tf.maximum(z_conv8_bn, tf.multiply(alpha, z_conv8_bn))

    # conv (NIN)
    W_conv8N = weight_variable([1, 1, 1, 512, 256])
    z_conv8N = conv3d(h_conv8, W_conv8N)
    batch_mean, batch_var = tf.nn.moments(z_conv8N, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(256))
    gamma = tf.Variable(tf.ones(256))
    z_conv8N_bn = tf.nn.batch_normalization(z_conv8N, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv8N = tf.maximum(z_conv8N_bn, tf.multiply(alpha, z_conv8N_bn))

    # flatten
    h_flat = tf.reshape(h_conv8N, shape=[-1, s * s * s * 256])

    # fc
    W_fc1 = weight_variable([s * s * s * 256, 512])
    z_fc1 = tf.matmul(h_flat, W_fc1)
    batch_mean, batch_var = tf.nn.moments(z_fc1, [0])
    beta = tf.Variable(tf.zeros(512))
    gamma = tf.Variable(tf.ones(512))
    z_fc1_bn = tf.nn.batch_normalization(z_fc1, batch_mean, batch_var, beta, gamma, 1e-5)
    h_fc1 = tf.maximum(z_fc1_bn, tf.multiply(alpha, z_fc1_bn))
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output
    W_score = weight_variable([512, s * s * s * (7 * num_bb + num_classes)])
    b_score = bias_variable([s * s * s * (7 * num_bb + num_classes)])
    pred = tf.add(tf.matmul(h_fc1_drop, W_score), b_score, name='pred')

    return pred


def yolo_vgg(data, s, num_bb, num_classes, keep_prob):

    # note: this network has s = 4, based on input size (32,256,256) and number of max pools
    assert(s == 4)

    # slope of leaky ReLU
    alpha = tf.constant(0.1)

    # conv
    W_conv0 = weight_variable([5, 5, 5, 1, 16])
    z_conv0 = conv3d(data, W_conv0)
    batch_mean, batch_var = tf.nn.moments(z_conv0, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv0_bn = tf.nn.batch_normalization(z_conv0, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv0 = tf.maximum(z_conv0_bn, tf.multiply(alpha, z_conv0_bn))

    # pool
    h_pool0 = pool2d(h_conv0)

    # conv
    W_conv2 = weight_variable([3, 3, 3, 16, 16])
    z_conv2 = conv3d(h_pool0, W_conv2)
    batch_mean, batch_var = tf.nn.moments(z_conv2, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv2_bn = tf.nn.batch_normalization(z_conv2, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv2 = tf.maximum(z_conv2_bn, tf.multiply(alpha, z_conv2_bn))

    # conv
    W_conv3 = weight_variable([3, 3, 3, 16, 16])
    z_conv3 = conv3d(h_conv2, W_conv3)
    batch_mean, batch_var = tf.nn.moments(z_conv3, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv3_bn = tf.nn.batch_normalization(z_conv3, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv3 = tf.maximum(z_conv3_bn, tf.multiply(alpha, z_conv3_bn))

    # pool
    h_pool1 = pool2d(h_conv3)

    # conv
    W_conv4 = weight_variable([3, 3, 3, 16, 32])
    z_conv4 = conv3d(h_pool1, W_conv4)
    batch_mean, batch_var = tf.nn.moments(z_conv4, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv4_bn = tf.nn.batch_normalization(z_conv4, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv4 = tf.maximum(z_conv4_bn, tf.multiply(alpha, z_conv4_bn))

    # conv
    W_conv5 = weight_variable([3, 3, 3, 32, 32])
    z_conv5 = conv3d(h_conv4, W_conv5)
    batch_mean, batch_var = tf.nn.moments(z_conv5, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv5_bn = tf.nn.batch_normalization(z_conv5, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv5 = tf.maximum(z_conv5_bn, tf.multiply(alpha, z_conv5_bn))

    # pool
    h_pool2 = pool2d(h_conv5)

    # conv
    W_conv6 = weight_variable([3, 3, 3, 32, 32])
    z_conv6 = conv3d(h_pool2, W_conv6)
    batch_mean, batch_var = tf.nn.moments(z_conv6, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv6_bn = tf.nn.batch_normalization(z_conv6, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv6 = tf.maximum(z_conv6_bn, tf.multiply(alpha, z_conv6_bn))

    # conv
    W_conv7 = weight_variable([3, 3, 3, 32, 32])
    z_conv7 = conv3d(h_conv6, W_conv7)
    batch_mean, batch_var = tf.nn.moments(z_conv7, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv7_bn = tf.nn.batch_normalization(z_conv7, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv7 = tf.maximum(z_conv7_bn, tf.multiply(alpha, z_conv7_bn))

    # pool
    h_pool3 = pool3d(h_conv7)

    # conv
    W_conv8 = weight_variable([3, 3, 3, 32, 64])
    z_conv8 = conv3d(h_pool3, W_conv8)
    batch_mean, batch_var = tf.nn.moments(z_conv8, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(64))
    gamma = tf.Variable(tf.ones(64))
    z_conv8_bn = tf.nn.batch_normalization(z_conv8, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv8 = tf.maximum(z_conv8_bn, tf.multiply(alpha, z_conv8_bn))

    # conv
    W_conv9 = weight_variable([3, 3, 3, 64, 64])
    z_conv9 = conv3d(h_conv8, W_conv9)
    batch_mean, batch_var = tf.nn.moments(z_conv9, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(64))
    gamma = tf.Variable(tf.ones(64))
    z_conv9_bn = tf.nn.batch_normalization(z_conv9, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv9 = tf.maximum(z_conv9_bn, tf.multiply(alpha, z_conv9_bn))

    # pool
    h_pool4 = pool3d(h_conv9)

    # conv
    W_conv10 = weight_variable([3, 3, 3, 64, 128])
    z_conv10 = conv3d(h_pool4, W_conv10)
    batch_mean, batch_var = tf.nn.moments(z_conv10, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(128))
    gamma = tf.Variable(tf.ones(128))
    z_conv10_bn = tf.nn.batch_normalization(z_conv10, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv10 = tf.maximum(z_conv10_bn, tf.multiply(alpha, z_conv10_bn))

    # conv
    W_conv11 = weight_variable([3, 3, 3, 128, 128])
    z_conv11 = conv3d(h_conv10, W_conv11)
    batch_mean, batch_var = tf.nn.moments(z_conv11, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(128))
    gamma = tf.Variable(tf.ones(128))
    z_conv11_bn = tf.nn.batch_normalization(z_conv11, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv11 = tf.maximum(z_conv11_bn, tf.multiply(alpha, z_conv11_bn))

    # pool
    h_pool5 = pool3d(h_conv11)

    # conv
    W_conv12 = weight_variable([3, 3, 3, 128, 256])
    z_conv12 = conv3d(h_pool5, W_conv12)
    batch_mean, batch_var = tf.nn.moments(z_conv12, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(256))
    gamma = tf.Variable(tf.ones(256))
    z_conv12_bn = tf.nn.batch_normalization(z_conv12, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv12 = tf.maximum(z_conv12_bn, tf.multiply(alpha, z_conv12_bn))

    # conv
    W_conv13 = weight_variable([3, 3, 3, 256, 512])
    z_conv13 = conv3d(h_conv12, W_conv13)
    batch_mean, batch_var = tf.nn.moments(z_conv13, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(512))
    gamma = tf.Variable(tf.ones(512))
    z_conv13_bn = tf.nn.batch_normalization(z_conv13, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv13 = tf.maximum(z_conv13_bn, tf.multiply(alpha, z_conv13_bn))

    # conv (NIN)
    W_conv14 = weight_variable([1, 1, 1, 512, 256])
    z_conv14 = conv3d(h_conv13, W_conv14)
    batch_mean, batch_var = tf.nn.moments(z_conv14, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(256))
    gamma = tf.Variable(tf.ones(256))
    z_conv14_bn = tf.nn.batch_normalization(z_conv14, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv14 = tf.maximum(z_conv14_bn, tf.multiply(alpha, z_conv14_bn))

    # flatten
    h_flat = tf.reshape(h_conv14, shape=[-1, s * s * s * 256])

    W_fc1 = weight_variable([s * s * s * 256, 512])
    z_fc1 = tf.matmul(h_flat, W_fc1)
    batch_mean, batch_var = tf.nn.moments(z_fc1, [0])
    beta = tf.Variable(tf.zeros(512))
    gamma = tf.Variable(tf.ones(512))
    z_fc1_bn = tf.nn.batch_normalization(z_fc1, batch_mean, batch_var, beta, gamma, 1e-5)
    h_fc1 = tf.maximum(z_fc1_bn, tf.multiply(alpha, z_fc1_bn))
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output
    W_score = weight_variable([512, s * s * s * (7 * num_bb + num_classes)])
    b_score = bias_variable([s * s * s * (7 * num_bb + num_classes)])
    pred = tf.add(tf.matmul(h_fc1_drop, W_score), b_score, name='pred')

    return pred


def yolo_vgg_ic(data, s, num_bb, num_classes, keep_prob):

    # note: this network has s = 4, based on input size (32,256,256) and number of max pools
    assert(s == 4)

    # slope of leaky ReLU
    alpha = tf.constant(0.1)

    # conv
    W_conv0 = weight_variable([5, 5, 5, 1, 16])
    z_conv0 = conv3d(data, W_conv0)
    batch_mean, batch_var = tf.nn.moments(z_conv0, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv0_bn = tf.nn.batch_normalization(z_conv0, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv0 = tf.maximum(z_conv0_bn, tf.multiply(alpha, z_conv0_bn))

    # pool
    h_pool0 = pool2d(h_conv0)

    # conv
    W_conv2 = weight_variable([3, 3, 3, 16, 16])
    z_conv2 = conv3d(h_pool0, W_conv2)
    batch_mean, batch_var = tf.nn.moments(z_conv2, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv2_bn = tf.nn.batch_normalization(z_conv2, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv2 = tf.maximum(z_conv2_bn, tf.multiply(alpha, z_conv2_bn))

    # conv
    W_conv3 = weight_variable([3, 3, 3, 16, 16])
    z_conv3 = conv3d(h_conv2, W_conv3)
    batch_mean, batch_var = tf.nn.moments(z_conv3, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(16))
    gamma = tf.Variable(tf.ones(16))
    z_conv3_bn = tf.nn.batch_normalization(z_conv3, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv3 = tf.maximum(z_conv3_bn, tf.multiply(alpha, z_conv3_bn))

    # pool
    h_pool1 = pool2d(h_conv3)

    # conv
    W_conv4 = weight_variable([3, 3, 3, 16, 32])
    z_conv4 = conv3d(h_pool1, W_conv4)
    batch_mean, batch_var = tf.nn.moments(z_conv4, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv4_bn = tf.nn.batch_normalization(z_conv4, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv4 = tf.maximum(z_conv4_bn, tf.multiply(alpha, z_conv4_bn))

    # conv
    W_conv5 = weight_variable([3, 3, 3, 32, 32])
    z_conv5 = conv3d(h_conv4, W_conv5)
    batch_mean, batch_var = tf.nn.moments(z_conv5, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv5_bn = tf.nn.batch_normalization(z_conv5, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv5 = tf.maximum(z_conv5_bn, tf.multiply(alpha, z_conv5_bn))

    # pool
    h_pool2 = pool2d(h_conv5)

    # conv
    W_conv6 = weight_variable([3, 3, 3, 32, 32])
    z_conv6 = conv3d(h_pool2, W_conv6)
    batch_mean, batch_var = tf.nn.moments(z_conv6, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv6_bn = tf.nn.batch_normalization(z_conv6, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv6 = tf.maximum(z_conv6_bn, tf.multiply(alpha, z_conv6_bn))

    # conv
    W_conv7 = weight_variable([3, 3, 3, 32, 32])
    z_conv7 = conv3d(h_conv6, W_conv7)
    batch_mean, batch_var = tf.nn.moments(z_conv7, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(32))
    gamma = tf.Variable(tf.ones(32))
    z_conv7_bn = tf.nn.batch_normalization(z_conv7, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv7 = tf.maximum(z_conv7_bn, tf.multiply(alpha, z_conv7_bn))

    # pool
    h_pool3 = pool3d(h_conv7)

    # conv
    W_conv8 = weight_variable([3, 3, 3, 32, 64])
    z_conv8 = conv3d(h_pool3, W_conv8)
    batch_mean, batch_var = tf.nn.moments(z_conv8, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(64))
    gamma = tf.Variable(tf.ones(64))
    z_conv8_bn = tf.nn.batch_normalization(z_conv8, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv8 = tf.maximum(z_conv8_bn, tf.multiply(alpha, z_conv8_bn))

    # conv
    W_conv9 = weight_variable([3, 3, 3, 64, 64])
    z_conv9 = conv3d(h_conv8, W_conv9)
    batch_mean, batch_var = tf.nn.moments(z_conv9, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(64))
    gamma = tf.Variable(tf.ones(64))
    z_conv9_bn = tf.nn.batch_normalization(z_conv9, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv9 = tf.maximum(z_conv9_bn, tf.multiply(alpha, z_conv9_bn))

    # pool
    h_pool4 = pool3d(h_conv9)

    # conv
    W_conv10 = weight_variable([3, 3, 3, 64, 128])
    z_conv10 = conv3d(h_pool4, W_conv10)
    batch_mean, batch_var = tf.nn.moments(z_conv10, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(128))
    gamma = tf.Variable(tf.ones(128))
    z_conv10_bn = tf.nn.batch_normalization(z_conv10, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv10 = tf.maximum(z_conv10_bn, tf.multiply(alpha, z_conv10_bn))

    # conv
    W_conv11 = weight_variable([3, 3, 3, 128, 128])
    z_conv11 = conv3d(h_conv10, W_conv11)
    batch_mean, batch_var = tf.nn.moments(z_conv11, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(128))
    gamma = tf.Variable(tf.ones(128))
    z_conv11_bn = tf.nn.batch_normalization(z_conv11, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv11 = tf.maximum(z_conv11_bn, tf.multiply(alpha, z_conv11_bn))

    # pool
    h_pool5 = pool3d(h_conv11)

    # conv
    W_conv12 = weight_variable([3, 3, 3, 128, 256])
    z_conv12 = conv3d(h_pool5, W_conv12)
    batch_mean, batch_var = tf.nn.moments(z_conv12, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(256))
    gamma = tf.Variable(tf.ones(256))
    z_conv12_bn = tf.nn.batch_normalization(z_conv12, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv12 = tf.maximum(z_conv12_bn, tf.multiply(alpha, z_conv12_bn))

    # conv
    W_conv13 = weight_variable([3, 3, 3, 256, 512])
    z_conv13 = conv3d(h_conv12, W_conv13)
    batch_mean, batch_var = tf.nn.moments(z_conv13, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(512))
    gamma = tf.Variable(tf.ones(512))
    z_conv13_bn = tf.nn.batch_normalization(z_conv13, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv13 = tf.maximum(z_conv13_bn, tf.multiply(alpha, z_conv13_bn))

    # conv (NIN)
    W_conv14 = weight_variable([1, 1, 1, 512, 256])
    z_conv14 = conv3d(h_conv13, W_conv14)
    batch_mean, batch_var = tf.nn.moments(z_conv14, [0, 1, 2, 3])
    beta = tf.Variable(tf.zeros(256))
    gamma = tf.Variable(tf.ones(256))
    z_conv14_bn = tf.nn.batch_normalization(z_conv14, batch_mean, batch_var, beta, gamma, 1e-5)
    h_conv14 = tf.maximum(z_conv14_bn, tf.multiply(alpha, z_conv14_bn))

    # flatten
    h_flat = tf.reshape(h_conv14, shape=[-1, s * s * s * 256])

    W_fc1 = weight_variable([s * s * s * 256, 1024])
    z_fc1 = tf.matmul(h_flat, W_fc1)
    batch_mean, batch_var = tf.nn.moments(z_fc1, [0])
    beta = tf.Variable(tf.zeros(1024))
    gamma = tf.Variable(tf.ones(1024))
    z_fc1_bn = tf.nn.batch_normalization(z_fc1, batch_mean, batch_var, beta, gamma, 1e-5)
    h_fc1 = tf.maximum(z_fc1_bn, tf.multiply(alpha, z_fc1_bn))
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fc
    W_fc2 = weight_variable([1024, 1024])
    z_fc2 = tf.matmul(h_fc1_drop, W_fc2)
    batch_mean, batch_var = tf.nn.moments(z_fc2, [0])
    beta = tf.Variable(tf.zeros(1024))
    gamma = tf.Variable(tf.ones(1024))
    z_fc2_bn = tf.nn.batch_normalization(z_fc2, batch_mean, batch_var, beta, gamma, 1e-5)
    h_fc2 = tf.maximum(z_fc2_bn, tf.multiply(alpha, z_fc2_bn))
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # output
    W_score = weight_variable([1024, s * s * s * (7 * num_bb + num_classes)])
    b_score = bias_variable([s * s * s * (7 * num_bb + num_classes)])
    pred = tf.add(tf.matmul(h_fc2_drop, W_score), b_score, name='pred')

    return pred
