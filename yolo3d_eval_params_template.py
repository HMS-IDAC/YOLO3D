# set batchsize
eval_batch_size = 8
# dimensions of input image
imsize = (32, 256, 256)
nchannels = 1
# trained model
model_name = 'yolo_vgg'
model_dir = '/home/dlr16/files/CellBiology/IDAC/Dave_Richmond/WeiChung_Lee/Coulter/Experiments/tensorflow_3d/' \
                'trial_12_vgg16_contd'
iter_num = 100000
# parameters of model output
s = 4
num_bb = 2
num_classes = 1
# threshold for correct detection
conf_thresh = 0.2
# gpu
gpu_id = 0
# log params
out_dir = '.'
# paths to data and annotations
img_path_list = ['/ssd/Coulter/img_vol1/all_data/vol_0', '/ssd/Coulter/img_vol1/all_data/vol_2']
roi_path_list = ['/ssd/Coulter/img_vol1/all_rois_refactored/vol_0', '/ssd/Coulter/img_vol1/all_rois_refactored/vol_2']
