from typing import Dict, List
import numpy as np
from numpy.lib.type_check import imag
from config import Config as cfg
import os.path as op
import torch
from typing import Dict, List

class cls_bbox:
    def __init__(self, use_label):
        self.use_label = use_label
        self.train_control = True

    def calculate_bbox(self, pred_bboxes, gt_bboxes, categories, device):
        #   calculate bbox
        ious = []
        gt_box_index = dict()

        for index, gt_bbox in enumerate(gt_bboxes):
            gt_bbox_cpu = gt_bbox.to('cpu')
            gt_bbox_add = np.tile(gt_bbox_cpu,(pred_bboxes.shape[0], 1) )
            gt_bbox_add = torch.tensor(gt_bbox_add,dtype=torch.float32).to(device)
            pred_bboxes = pred_bboxes.to(device)

            lt = torch.max(pred_bboxes[..., :2], gt_bbox_add[..., :2])
            rb = torch.min(pred_bboxes[..., 2:], gt_bbox_add[..., 2:])

            wh = (rb-lt).clamp(min=0)
            inter = wh[:, 0] * wh[:, 1]

            area_a = torch.prod(pred_bboxes[..., 2:] - pred_bboxes[..., :2], 1)
            area_b = torch.prod(gt_bbox_add[..., 2:] - gt_bbox_add[..., :2], 1)

            gt_index = 'gt_{index}'
            gt_box_index[gt_index] = gt_bbox

            iou = inter / (area_a + area_b - inter + 1e-6)
            ious.append(iou)

        return ious, gt_box_index



    def bbox_bath(self, true_indices, use_indices, pred_bboxes, gt_bbox, device):
        true_bbox_indices = use_indices[true_indices]
        true_bboxes = pred_bboxes[true_bbox_indices]
        true_bboxes = true_bboxes.to('cpu').numpy()

        gt_box = gt_bbox.to('cpu').numpy()
        gt_bboxes = np.tile(gt_box, (true_bboxes.shape[0], 1))


        return true_bboxes.tolist(), gt_bboxes.tolist()

    def make_minibatch_(self, images:List, label_sets:Dict, pred_bboxes, gt_bboxes, device):
        image_list = []
        label_list = []
        true_bbox_list = []
        gt_bbox_list = []
        true_len = 0
        for i, label_set in enumerate(label_sets.values()):
            # print('labelsss', label_set)
            use_indices = label_set['Use_indices']
            labels = label_set['labels']
            true_num = label_set['true_num']
            true_len += true_num
            true_indices = np.where(labels > 0)
            select_img = self.choose_data(use_indices, images)

            # true_bboxes, gt_bboxes = self.bbox_bath(true_indices, use_indices, pred_bboxes, gt_bboxes[i], device)

            image_list += select_img
            label_list += list(labels)

            # true_bbox_list += true_bboxes
            # gt_bbox_list += gt_bboxes

        # bbox 만들어야함

        # true_bbox_list = torch.tensor(true_bbox_list).to(device)
        # gt_bbox_list = torch.tensor(gt_bbox_list).to(device)

        # return image_list, label_list, true_bbox_list, gt_bbox_list
        return image_list, label_list, true_len

    def choose_data(self, use_indices, images):
        select_img = []
        for index in use_indices:
            select_img.append(images[index])

        return select_img


    def convert_xyxy_to_xywh(self, pred_bbox, gt_bbox):

        # Calculate x/y/w/h of P/G
        # predicted box width, height, centerX coord, centerY coord
        p_w = pred_bbox[:, 2] - pred_bbox[:, 0]
        p_h = pred_bbox[:, 1] - pred_bbox[:, 3]
        p_x = pred_bbox[:, 0] + p_w / 2
        p_y = pred_bbox[:, 1] + p_h / 2

        # ground truth box width, height, center x , center y
        g_w = gt_bbox[:, 2] - gt_bbox[:, 0]
        g_h = gt_bbox[:, 1] - gt_bbox[:, 3]
        g_x = gt_bbox[:, 0] + g_w / 2
        g_y = gt_bbox[:, 1] + g_h / 2

        t_x = (g_x - p_x) / p_w
        t_y = (g_y - p_y) / p_y
        t_w = torch.log(g_w / p_w + 1e-6)
        t_h = torch.log(g_h / p_h + 1e-6)

        t_x = t_x.reshape((-1, 1))
        t_y = t_y.reshape((-1, 1))
        t_w = t_w.reshape((-1, 1))
        t_h = t_h.reshape((-1, 1))

        # bbox_t = torch.tensor((t_x, t_y, t_w, t_h), dtype=torch.float32).to(device)
        bbox_t = torch.stack([t_x.T, t_y.T, t_w.T, t_h.T]).T.reshape(-1, 4)
        # bbox_t = bbox_t * gt_mask

        return bbox_t

    def sort_iou(self, iou, category, device):
        state = True
        sorted_iou, indices = torch.sort(iou, descending=True)

        True_indices = torch.where(sorted_iou >= 0.5)
        Negative_indices = torch.where(sorted_iou < 0.5)
        True_ious_len = len(True_indices[0])
        Negative_len = True_ious_len*7

        Negative_indices = Negative_indices[0][:Negative_len]

        True_ious = indices[True_indices]
        Negative_ious = indices[Negative_indices]

        labels = np.zeros(True_ious_len + Negative_len)
        labels[:True_ious_len] = 1
        labels = labels * category.to('cpu').numpy()
        Use_indices = torch.cat((True_ious, Negative_ious))

        return Use_indices, labels, True_ious_len

    def set_Clslabel(self, images, ious, categories, pred_bboxes, gt_bboxes, gt_box_index, device):
        label_sets = {}
        # print('ious',ious)
        for i, iou in enumerate(ious):
            Use_indices, labels, true_num = self.sort_iou(iou, categories[i], device)
            # negative_minibatch_size = cfg.Train_set.mini_batch_size - len(True_ious)
            # print('True_ious:', True_ious)
            cate = str('label'+str(i))
            label_sets[cate] = {'Use_indices': Use_indices, 'labels': labels, 'true_num':true_num, 'category': categories[i], 'gt_box_index': i+1}

        # images, labels, true_bbox_list, gt_bbox_list = self.make_minibatch_(images, label_sets, pred_bboxes, gt_bboxes, device)
        images, labels, true_len = self.make_minibatch_(images, label_sets, pred_bboxes, gt_bboxes, device)


        # target_bboxes = self.convert_xyxy_to_xywh(true_bbox_list, gt_bbox_list)
        # labels = torch.tensor(labels, dtype=torch.uint8).to(device)
        labels = torch.tensor(labels)

        labels = labels.type(torch.LongTensor).to(device).reshape(-1)

        # label_len = len(target_bboxes)
        # return images, labels, target_bboxes, label_len
        return images, labels, true_len


    def __call__(self, images, pred_bboxes, targets, device):
        categories = targets['category']
        gt_bboxes = targets['bboxes']

        pred_bboxes = torch.tensor(pred_bboxes, dtype=torch.float32).reshape((len(pred_bboxes), 4)).to(device)
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32).to(device)

        ious, gt_box_index = self.calculate_bbox(pred_bboxes, gt_bboxes, categories, device)
        # images, labels, target_bbox, label_len = self.set_Clslabel(images, ious, categories, pred_bboxes, gt_bboxes, gt_box_index, device)
        images, labels, true_len = self.set_Clslabel(images, ious, categories, pred_bboxes, gt_bboxes, gt_box_index, device)

        return images, labels, true_len


# if __name__ == '__main__':
#     LABELPATH = op.join(cfg.SRCPATH, 'label_2')
#     cls = cls_bbox(LABELPATH)
#     cls(2)
