# import torch
#
# def NMS(self, bboxes, confidence, iou_thresold, threshold):
#     assert type(bboxes) == list
#
#     #   Temporarily save
#     # ious = iou_function(bboxes)
#
#     #   if bbox consit [class_index, clas_score, x1, y1, x2, y2]
#     bboxes = [bbox for bbox in bboxes if bbox[1] > threshold]
#     #   sort bboxes
#     bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
#     bboxes_after_nmn = []
#
#     while bboxes:
#
#         chosen_bbox = bboxes.pop(0)
#         bboxes = [box for box in bboxes if box[0] != chosen_bbox[0] \
#                   or ious < iou_thresold]
#
#         bboxes_after_nmn.append(chosen_bbox)
#
#     return  bboxes_after_nmn
#


import torch
from torchvision.ops import nms
def NMS(bboxes, class_list, confidence, iou_thresold, threshold):
    assert type(bboxes) == list
    out_box = []
    out_class = []

    bbox_tensor = torch.tensor(bboxes, dtype=torch.float32).reshape((-1, 4))
    class_tensor = torch.tensor(class_list)
    confidence = torch.tensor(confidence).reshape((-1))
    select_candidate = nms(bbox_tensor, confidence, 0.7)
    print(select_candidate)

    for index in select_candidate.numpy():
        out_box.append(bboxes[index])
        out_class.append(class_list[index])

    return out_box, out_class
    # #   Temporarily save
    # # ious = iou_function(bboxes)
    #
    # #   if bbox consit [class_index, clas_score, x1, y1, x2, y2]
    # bboxes = [bbox for bbox in bboxes if bbox[1] > threshold]
    # #   sort bboxes
    # bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    # bboxes_after_nmn = []
    #
    # while bboxes:
    #
    #     chosen_bbox = bboxes.pop(0)
    #     bboxes = [box for box in bboxes if box[0] != chosen_bbox[0] \
    #               or ious < iou_thresold]
    #
    #     bboxes_after_nmn.append(chosen_bbox)

    return  bboxes_after_nmn



