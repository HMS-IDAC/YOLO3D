# set batch size
train_batch_size = 8
val_batch_size = 8
# dimensions of input image
imsize = (32, 256, 256)
nchannels = 1
# model name
model_name = 'yolo_vgg'
# parameters of model output
s = 4
num_bb = 2  # this currently can't be changed
num_classes = 1
# solver params
n_iters = 1000000
base_lr = 1e-3
dropout_keep_prob = 0.9
frac_pos = 0.75  # this sets the fraction of images that contain at least one object
# log params
log_interval = 50
val_interval = 50
save_interval = 2e4
out_dir = '.'
# gpu
gpu_id = 3
# initialization
init_model = 'random'
init_iters = 0
# paths to data and annotations
img_path_list = ['/ssd/Coulter/img_vol1/all_data/vol_0', '/ssd/Coulter/img_vol1/all_data/vol_2']
roi_path_list = ['/ssd/Coulter/img_vol1/all_rois_refactored/vol_0', '/ssd/Coulter/img_vol1/all_rois_refactored/vol_2']
# split up the annotated volume, to avoid cross-talk of negative samples.  format=[x1, x2, y1, y2, z1, z2]
train_img_coords_list = [[0, 8832, 0, 5120, 0, 100], [0, 8832, 0, 5120, 0, 100]]
val_img_coords_list = [[8832, 11776, 0, 2560, 0, 100], [8832, 11776, 0, 2560, 0, 100]]
test_img_coords_list = [[8832, 11776, 2560, 5120, 0, 100], [8832, 11776, 2560, 5120, 0, 100]]
