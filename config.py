import os.path as op
import numpy as np

class Config:
    # SRCPATH = '/media/milab/My Passport3/toV100/dongun/Experiment_1/total_data'
    SRCPATH = '/home/user/hdd/datasets/kitti'
    # A2D2 = '/media/milab/My Passport3/toV100/A2D2/camera_lidar/20180810_150607'
    # A2IMG = op.join(A2D2,'camera')
    # A2LD = op.join(A2D2,'lidar', 'cam_front_center')
    # VELOPATH = op.join(SRCPATH,'velo','training', 'velodyne')
    # IMGPATH = op.join(SRCPATH,'img', 'training','image_2')
    # CALPATH = '/home/milab/machine_ws/experiment/calibration/training/calib'
    VELOPATH = op.join(SRCPATH,'training','velodyne')
    IMGPATH = op.join(SRCPATH,'training','image_2')
    LABELPATH = op.join(SRCPATH,'training','label_2')
    CALPATH = op.join(SRCPATH,'training', 'calib')

    class Lidar_set:
        # resize_list = [[-5,-20, 5, 20], [-20, 5, 20, 5],[-10,-10, 10,10], [-20,-20, 20,20], [-30,-30, 30,30]]
        # resize_list = [[0, 0, 0, 0], [-5,-20, 5, 20], [-20, 5, 20, 5],[-10,-10, 10,10], [-13, -13, 13, 13], [-20,-20, 20,20]]
        resize_list = [[0, 0, 0, 0], [-13, -13, 13, 13], [-15, -15, 15, 15]]

    class Train_set:
        use_label = ['Car', 'Pedestrian', 'Truck']
        label_index = {'Background': 0, 'Car': 1, 'Pedestrian': 2, 'Truck': 3}
        index_to_label = {'0': 'Background', '1': 'Car', '2': 'Pedestrian', '3': 'Truck'}
        mini_batch_size = 20
