import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.type_check import imag
import open3d as o3d
from config import Config as cfg
import os.path as op
from PIL import Image
# import cv2
import matplotlib.pyplot as plt
from kitti_util import Calibration
from transformation import Transformation
from torchvision import transforms


# import torch


class LidarCluster:
    def projection_img(self, images, pcd, cal, labels):
        # file = '{0:06d}.png' .format(index)
        pts = np.asarray(pcd.points)
        bottom = np.ones((pts.shape[0], 1))

        pts = np.concatenate((pts, bottom), axis=1)

        V2C = cal.V2C
        v2c = np.vstack((V2C, [0, 0, 0, 1]))

        pts_c = np.dot(v2c, pts.T)
        pts_v = pts_c[:3, :]
        cal.P = cal.P[:3, :3]

        projection = np.dot(cal.P, pts_v)
        projection = projection / projection[2, :]

        proj = projection[:2, :]
        image_n = np.array(images)
        img_shape = image_n.shape
        # print('img shape:', img_shape)
        img = images
        # print('img shape:', img_shape)
        # tf = transforms.ToPILImage()
        # img = tf(images)

        img_n = np.array(img)

        zeros = np.zeros((img_shape[0], img_shape[1]))
        # print(zeros.shape)
        for i in (range(proj.shape[1])):
            x = int(proj[0][i])
            y = int(proj[1][i])
            # print(x,y)
            if (0 < y < img_shape[0]) and (0 < x < img_shape[1]):
                zeros[y, x] = labels[i]
        zeros = zeros * 10

        index = np.unique(zeros)
        # print(f'split:{len(index)}',index, )
        bboxes = []
        images = []
        for i in index:
            if i > 0:
                black = np.zeros((img_n.shape[0], img_n.shape[1]))
                # print('unique index:', index)
                split_index = np.where(zeros == i)
                # tf224 = Transformation(224)
                resize_list = cfg.Lidar_set.resize_list
                for (a, b, c, d) in resize_list:
                    cluster_target = {}

                    y1, x1, y2, x2 = np.min(split_index[0]) + a, np.min(split_index[1]) + b, np.max(
                        split_index[0]) + c, np.max(split_index[1]) + d
                    crop_axis_x = np.array([x1, x2])
                    crop_axis_y = np.array([y1, y2])
                    crop_axis_x = np.clip(crop_axis_x, 0, img_n.shape[1])
                    crop_axis_y = np.clip(crop_axis_y, 0, img_n.shape[0])
                    crop_axis = np.array([crop_axis_x[0], crop_axis_y[0], crop_axis_x[1], crop_axis_y[1]])
                    crop_img = img.crop((crop_axis))
                    # img = np.array(img)
                    # crop_img = img[crop_axis[0]:crop_axis[2], crop_axis[1]:crop_axis[3],: ]
                    # print(crop_img)
                    black[split_index] = i
                    #   trnasform image
                    # print('test:',crop_img.size)
                    # print('img shape:', np.array(crop_img).shape)
                    # try:
                    #     # image = tf224(crop_img)
                    #     # plt.imshow(crop_img)
                    #     # plt.show()
                    #     images.append(crop_img)
                    #     bboxes.append([crop_axis])
                    # except:
                    #     continue
                    if (crop_img.size[0] != 0) and (crop_img.size[1] != 0):
                        # image = tf224(crop_img)
                        # plt.imshow(crop_img)
                        # plt.show()
                        images.append(crop_img)
                        bboxes.append([crop_axis])

                    # print('bbbb:', bboxes)
                    # select_region = Propose_region(images, bboxes)
                    # plt.imshow(crop_img)
                    # plt.show()
        return images, bboxes

    def preprocess(self, points):
        x_range, y_range, z_range = (3, 70), (-15, 15), (-1.3, 2.5)
        # print(points.shape)
        points1 = points[np.logical_and.reduce((points[:, 0] > x_range[0], points[:, 0] < x_range[1], \
                                                points[:, 1] > y_range[0], points[:, 1] < y_range[1], \
                                                points[:, 2] > z_range[0], points[:, 2] < z_range[1]))]

        eps_ = 10
        m_points = 4
        z_axis = points1[:, 2].copy() * 50

        points1[:, 2] = 0
        points1 = points1 * 50
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points1)
        # pcd = pcd.voxel_down_sample(voxel_size=0.2)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            # print('cluster: ',len(pcd.cluster_dbscan(eps=eps_, min_points=m_points)))
            labels = np.array(
                pcd.cluster_dbscan(eps=eps_, min_points=m_points, print_progress=False))

        max_label = labels.max()
        # print(f"point cloud has {max_label+1} clusters")

        colors1 = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors1[labels < 0] = 0

        pcd.colors = o3d.utility.Vector3dVector(colors1[:, :3])
        target_pcd = np.asarray(pcd.points)
        # target = np.array(target_pcd)

        target_pcd[:, 2] = z_axis
        # print('target:', target_pcd)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target_pcd)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd.colors = o3d.utility.Vector3dVector(colors1[:, :3])

        return pcd, labels

    def view_points(self, pts):
        # zeros = np.zeros((proj.shape[1])).reshape((1, -1))
        # pts = np.concatenate((proj, zeros), axis=0)
        pts = pts.T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size=0.9)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        # print("bbox:", x)
        eps_ = 0.5
        m_points = 10
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            # print('cluster: ',len(pcd.cluster_dbscan(eps=eps_, min_points=m_points)))
            labels = np.array(
                pcd.cluster_dbscan(eps=eps_, min_points=m_points, print_progress=False))

        max_label = labels.max()
        # print(f"point cloud has {max_label+1} clusters")

        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # o3d.visualization.draw_geometries([pcd],
        #                                 zoom=0.7,
        #                                 front=[0.5439, -0.2333, -0.8060],
        #                                 lookat=[2.4615, 2.1331, 1.338],
        #                                 up=[-0.1781, -0.9708, 0.1608])
    def load_gt_bbox(self, targets):
        with open(path, 'r') as r:
            anns = r.readlines()
        bboxes = []
        category = []
        for ann in anns:
            ann = ann.strip('\n').split(' ')
            # print(ann)
            if ann[0] in self.use_label:
                category.append(ann[0])
                bboxes.append([float(ann[4]), float(ann[5]), float(ann[6]), float(ann[7])])
        # print(bbox)
        return category, bboxes



    def __call__(self, images, lidar, targets, cal):

        points = lidar['points']

        pcd, labels = self.preprocess(points)
        images, bboxes = self.projection_img(images, pcd, cal, labels)
        check = len(np.unique(labels))
        return images, bboxes, check

# def main(index):

#     velo_index = '{0:06d}.bin' .format(index)
#     cal_index = '{0:06d}.txt' .format(index)
#     cal_path = op.join(cfg.CALPATH, cal_index)
#     cal = Calibration(cal_path)
#     points = np.fromfile(op.join(cfg.VELOPATH, velo_index), dtype=np.float32).reshape(-1, 4)
#     # points = np.asarray()
#     intensity = points[:, 3]
#     # print(intensity.shape)
#     # z_axis = points[:, 2]
#     points = points[:, 0:3]

#     pcd, labels = preprocess(points)
#     images, bboxes = projection_img(index, pcd, cal, labels)
#     # view_points(proj)


# if __name__ == "__main__":
#     for index in range(200):
#         main(index)
