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
            inter = wh[:,0] * wh[:, 1]

            area_a = torch.prod(pred_bboxes[..., 2:] - pred_bboxes[..., :2], 1)
            area_b = torch.prod(gt_bbox_add[..., 2:] - gt_bbox_add[..., :2], 1)

            gt_index = 'gt_{index}'
            gt_box_index[gt_index] = gt_bbox

            iou = inter / (area_a + area_b - inter + 1e-6)
            ious.append(iou)

        return ious, gt_box_index


    def load_gt_bbox(self, path):
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

    # def Negativesort(self, negative, ious, size):
    #     ious_n = ious[negative]

    #     ious_cat = dict()
    #     for index, iou in enumerate(ious_n):
    #         ious_cat[f'{index}'] = iou
    #     ious_sort = sorted(ious_cat.items(), key=(lambda x:x[1]), reverse=True)
    #     split = ious_sort[:15]
    #     negative_index = [x for x, score in split]
    #     print('negative:', negative_index)

    def Negativesort(self,images, label_sets, device):
        mini_batch = []
        labels = torch.zeros(len(images)).to(device)

    # def make_minibatch(self, images:List, label_sets:Dict, device):
    #     mini_batch = []
    #     labels = torch.zeros(len(images)).to(device)
    #     # print('len: ', images)
    #     for  label_set in label_sets.values():
    #         # print('labelsss', label_set)
    #         True_set_index =  label_set['True'][0]
    #         # Negative_set_index =  label_set['Negative'][0]
    #         labels[True_set_index] = label_set['category']
    #         # print('labels:', labels)
    #     # print(labels)
    #     # print('@@@@@@@@@@labels unique:', torch.unique(labels))
    #     label_len = len(torch.unique(labels))

    #     return images, labels, label_len

    def stack_data(self, true_img, true_label, neg_img, neg_label):
        cat_img = true_img + neg_img
        cat_label = true_label + neg_label
        print('cat:',len(cat_img))
        print(cat_label)

    def set_minibatch(self, images, labels, pred_bboxes, gt_box_labels):
        true_images = []
        true_labels = []
        true_bboxes = []
        true_gt_labels = []

        negative_images = []
        negative_labels = []
        negative_bboxes = []
        negative_gt_labels = []

        for i, label in enumerate(labels):
            # print('done labels:', label)
            if label.to('cpu').numpy() == 0:
                negative_images.append(images[i])
                negative_labels.append(label)
                negative_bboxes.append(pred_bboxes[i])
                negative_gt_labels.append(gt_box_labels[i])

            elif label.to('cpu').numpy() != 0:
                true_images.append(images[i])
                true_labels.append(label)
                true_bboxes.append(pred_bboxes[i])
                true_gt_labels.append(gt_box_labels[i])

        if len(true_images) != 0:
            negative_len = len(true_images) * 3
        else:
            negative_len = 10

        negative_select_img = negative_images[: negative_len]
        negative_select_label = negative_labels[: negative_len]
        negative_select_bbox = negative_bboxes[: negative_len]
        negative_select_gtlabel = negative_gt_labels[: negative_len]

        cat_images = true_images + negative_select_img
        cat_labels = true_labels + negative_select_label
        cat_bboxes = true_bboxes + negative_select_bbox
        cat_gt_labels = true_gt_labels + negative_select_gtlabel

        return cat_images, cat_labels, cat_bboxes, cat_gt_labels

    def make_minibatch(self, images:List, label_sets:Dict, pred_bboxes, device):
        mini_batch = []
        labels = torch.zeros(len(images)).to(device)
        gt_box_labels = torch.zeros(len(images)).to(device)
        # print('len: ', images)
        for  label_set in label_sets.values():
            # print('labelsss', label_set)
            True_set_index =  label_set['True'][0]
            Negative_set_index =  label_set['Negative'][0]
            labels[True_set_index] = label_set['category']
            gt_box_labels[True_set_index] = label_set['gt_box_index']

        label_len = len(torch.unique(labels))
        images, labels, bboxes, gt_labels = self.set_minibatch(images, labels, pred_bboxes, gt_box_labels)

        return images, labels, bboxes, gt_labels, label_len

    def convert_xyxy_to_xywh(self, pred_bbox, gt_bbox, gt_mask, device):

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
        print(t_x)
        print(t_h)
        # bbox_t = torch.tensor((t_x, t_y, t_w, t_h), dtype=torch.float32).to(device)
        bbox_t = torch.stack([t_x.T, t_y.T, t_w.T, t_h.T]).T.reshape(-1, 4)
        bbox_t = bbox_t * gt_mask

        return bbox_t

    def set_Clslabel(self, images, ious, categories, pred_bboxes, gt_bboxes, gt_box_index, device):
        label_sets = {}
        # print('ious',ious)
        for i, iou in enumerate(ious):
            True_ious = torch.where(iou >= 0.5)
            Negative_ious = torch.where(iou < 0.5)
            # negative_minibatch_size = cfg.Train_set.mini_batch_size - len(True_ious)
            # print('True_ious:', True_ious)
            cate = str('label'+str(i))

            label_sets[cate] = {'True': True_ious,'Negative': Negative_ious,'category': categories[i], 'gt_box_index': i+1}
        # self.Negativesort(images, label_sets, device)
        images, labels, bboxes, gt_labels, label_len = self.make_minibatch(images, label_sets, pred_bboxes, device)
        pred_bbox, gt_bbox, gt_mask = self.bbox_reg_batch(gt_labels, bboxes, gt_bboxes, device)
        target_bbox = self.convert_xyxy_to_xywh(pred_bbox, gt_bbox, gt_mask, device)

        return images, labels, target_bbox, label_len

    # def bbox_reg_batch(self, labels_sets, pred_bbox, gt_bbox, device):
    #     Train = []
    #     Target = []
    #     for i, label in enumerate(labels_sets.values()):
    #         True_index = label['True'][0]
    #         # device = label['True'][0].device
    #         target_bbox = gt_bbox[i].to('cpu')
    #         pred_box_true = pred_bbox[True_index]
    #
    #         gt_bbox_target = np.tile(target_bbox, (len(True_index), 1))
    #         # gt_bbox_target = torch.from_numpy(gt_bbox_target)
    #
    #         # if len(True_index) != 0:
    #         Train.extend(pred_box_true)
    #         Target.extend(gt_bbox_target)
    #         # Train.append(pred_box_true)
    #         # Target.append(gt_bbox_target)
    #     # Train = Train.reshape((-1,4))
    #     # Target = torch.tensor(Target).to(device)
    #
    #     bbox_dataset = dict(Train_box=Train, Target_box=Target)
    #
    #     return bbox_dataset

    def bbox_reg_batch(self, gt_labels, pred_bbox, gt_bbox, device):
        gt_boxes = []

        false_bbox = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(device)
        for gt_label in gt_labels:
            index = int(gt_label - 1)
            if gt_label == 0:
                gt_boxes.append(false_bbox)
            else:
                gt_boxes.append(gt_bbox[index])

        gt_bboxes = torch.stack(gt_boxes)
        pred_bboxes = torch.stack(pred_bbox)

        gt_mask = torch.sum(gt_bboxes, 1).reshape((-1, 1))
        gt_mask = gt_mask.to('cpu')
        replace_value = torch.tensor(1., dtype=torch.float32)
        gt_mask = torch.where(gt_mask > 0., 1., 0.).to(device)

        return pred_bboxes, gt_bboxes, gt_mask



    def __call__(self, images, pred_bboxes, targets, device):
        categories = targets['category']
        gt_bboxes = targets['bboxes']

        pred_bboxes = torch.tensor(pred_bboxes, dtype=torch.float32).reshape((len(pred_bboxes), 4)).to(device)
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32).to(device)

        ious, gt_box_index  = self.calculate_bbox(pred_bboxes, gt_bboxes, categories, device)
        images, labels, target_bbox, label_len = self.set_Clslabel(images, ious, categories, pred_bboxes, gt_bboxes, gt_box_index, device)

        return images, labels, target_bbox, label_len
#
# def convert_xyxy_to_xywh(bbox_dataset, device):
#
#         trains = bbox_dataset['Train_box']
#         targets = bbox_dataset['Target_box']
#
#         # Calculate x/y/w/h of P/G
#         # predicted box width, height, centerX coord, centerY coord
#         p_w = trains[:, 2] - trains[:, 0]
#         p_h = trains[:, 1] - trains[:, 3]
#         p_x = trains[:, 0] + p_w / 2
#         p_y = trains[:, 1] + p_h / 2
#
#         # ground truth box width, height, center x , center y
#         g_w = targets[:, 2] - targets[:, 0]
#         g_h = targets[:, 1] - targets[:, 3]
#         g_x = targets[:, 0] + g_w / 2
#         g_y = targets[:, 1] + g_h / 2
#
#         t_x = (g_x - p_x) / p_w
#         t_y = (g_y - p_y) / p_y
#         t_w = torch.log(g_w / p_w)
#         t_h = torch.log(g_h / p_h)
#
#         t_x = t_x.reshape((-1, 1))
#         t_y = t_y.reshape((-1, 1))
#         t_w = t_w.reshape((-1, 1))
#         t_h = t_h.reshape((-1, 1))
#         print(t_x)
#         print(t_h)
#         # bbox_t = torch.tensor((t_x, t_y, t_w, t_h), dtype=torch.float32).to(device)
#         bbox_t = torch.stack([t_x.T, t_y.T, t_w.T, t_h.T]).T.reshape(-1, 4)
#
#         return bbox_t




# if __name__ == '__main__':
#     LABELPATH = op.join(cfg.SRCPATH, 'label_2')
#     cls = cls_bbox(LABELPATH)
#     cls(2)
