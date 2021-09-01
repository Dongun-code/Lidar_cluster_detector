import torch

def NMS(self, bboxes, confidence, iou_thresold, threshold):
    assert type(bboxes) == list

    #   Temporarily save
    # ious = iou_function(bboxes)

    #   if bbox consit [class_index, clas_score, x1, y1, x2, y2]
    bboxes = [bbox for bbox in bboxes if bbox[1] > threshold]
    #   sort bboxes
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nmn = []

    while bboxes:

        chosen_bbox = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[0] != chosen_bbox[0] \
                  or ious < iou_thresold]

        bboxes_after_nmn.append(chosen_bbox)

    return  bboxes_after_nmn



