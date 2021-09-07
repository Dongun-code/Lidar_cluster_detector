# import os.path as op
# import numpy as np
#
# class Config:
#     SRCPATH = '/media/milab/My Passport3/toV100/dongun/Experiment_1/total_data'
#     # SRCPATH = '/home/user/hdd/datasets/kitti'
#     # A2D2 = '/media/milab/My Passport3/toV100/A2D2/camera_lidar/20180810_150607'
#     # A2IMG = op.join(A2D2,'camera')
#     # A2LD = op.join(A2D2,'lidar', 'cam_front_center')
#     # VELOPATH = op.join(SRCPATH,'velo','training', 'velodyne')
#     # IMGPATH = op.join(SRCPATH,'img', 'training','image_2')
#     # CALPATH = '/home/milab/machine_ws/experiment/calibration/training/calib'
#     VELOPATH = op.join(SRCPATH,'training','velodyne')
#     IMGPATH = op.join(SRCPATH,'training','image_2')
#     LABELPATH = op.join(SRCPATH,'training','label_2')
#     CALPATH = op.join(SRCPATH,'training', 'calib')
#
#     R_ = np.array(
#         [7.755449e-03, -9.999694e-01, -1.014303e-03 ,
#             2.294056e-03 ,1.032122e-03, -9.999968e-01,
#              9.999673e-01, 7.753097e-03, 2.301990e-03]).reshape((3, 3))
#     T_ = np.array([-7.275538e-03, -6.324057e-02, -2.670414e-01])
#
#     RT = np.array(
#         [7.755449e-03, -9.999694e-01, -1.014303e-03 ,-7.275538e-03,
#             2.294056e-03 ,1.032122e-03, -9.999968e-01,-6.324057e-02,
#              9.999673e-01, 7.753097e-03, 2.301990e-03,-2.670414e-01,
#              0, 0,0 ,1]).reshape((4, 4))
#
#     P = np.array([7.183351e+02, 0.000000e+00, 6.003891e+02,
#      0.000000e+00, 7.183351e+02, 1.815122e+02,
#       0.000000e+00, 0.000000e+00 ,1.000000e+00]).reshape((3, 3))
#
#     class Lidar_set:
#         resize_list = [[-5,-20, 5, 20], [-20, 5, 20, 5],[-10,-10, 10,10], [-20,-20, 20,20], [-30,-30, 30,30]]
#
#     class Train_set:
#         use_label = ['Car', 'Pedestrian','Truck', 'Cyclist']
#         label_index = {'Background':0, 'Car':1, 'Pedestrian':2, 'Truck':3, 'Cyclist':4}
#         index_to_label = {'1': 'Car', '2':'Pedestrian', '3': 'Truck', '4':'Cyclist'}
#         mini_batch_size = 30

import os.path as op
import numpy as np


class Config:
    SRCPATH = '/media/milab/My Passport3/toV100/dongun/Experiment_1/total_data'
    # SRCPATH = '/home/user/hdd/datasets/kitti'
    # A2D2 = '/media/milab/My Passport3/toV100/A2D2/camera_lidar/20180810_150607'
    # A2IMG = op.join(A2D2,'camera')
    # A2LD = op.join(A2D2,'lidar', 'cam_front_center')
    # VELOPATH = op.join(SRCPATH,'velo','training', 'velodyne')
    # IMGPATH = op.join(SRCPATH,'img', 'training','image_2')
    # CALPATH = '/home/milab/machine_ws/experiment/calibration/training/calib'
    VELOPATH = op.join(SRCPATH,'velo', 'training', 'velodyne')
    IMGPATH = op.join(SRCPATH, 'img', 'training', 'image_2')
    LABELPATH = op.join(SRCPATH, 'label_2')
    CALPATH = op.join(SRCPATH, 'calibration', 'training', 'calib')

    R_ = np.array(
        [7.755449e-03, -9.999694e-01, -1.014303e-03,
         2.294056e-03, 1.032122e-03, -9.999968e-01,
         9.999673e-01, 7.753097e-03, 2.301990e-03]).reshape((3, 3))
    T_ = np.array([-7.275538e-03, -6.324057e-02, -2.670414e-01])

    RT = np.array(
        [7.755449e-03, -9.999694e-01, -1.014303e-03, -7.275538e-03,
         2.294056e-03, 1.032122e-03, -9.999968e-01, -6.324057e-02,
         9.999673e-01, 7.753097e-03, 2.301990e-03, -2.670414e-01,
         0, 0, 0, 1]).reshape((4, 4))

    P = np.array([7.183351e+02, 0.000000e+00, 6.003891e+02,
                  0.000000e+00, 7.183351e+02, 1.815122e+02,
                  0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape((3, 3))

    class Lidar_set:
        resize_list = [[-5, -20, 5, 20], [-20, 5, 20, 5], [-10, -10, 10, 10], [-20, -20, 20, 20], [-30, -30, 30, 30]]

    class Train_set:
        use_label = ['Car', 'Pedestrian', 'Truck', 'Cyclist']
        label_index = {'Background': 0, 'Car': 1, 'Pedestrian': 2, 'Truck': 3, 'Cyclist': 4}
        index_to_label = {'1': 'Car', '2': 'Pedestrian', '3': 'Truck', '4': 'Cyclist'}
        mini_batch_size = 30
