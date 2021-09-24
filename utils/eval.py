import numpy as np
from .result_io import load_gt_file

from typing import Optional


def score_iou(det_bbox, gt_bbox):
    ixmin = np.maximum(gt_bbox[0], det_bbox[0])
    iymin = np.maximum(gt_bbox[1], det_bbox[1])
    ixmax = np.minimum(gt_bbox[2], det_bbox[2])
    iymax = np.minimum(gt_bbox[3], det_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (det_bbox[2] - det_bbox[0] + 1.0) * (det_bbox[3] - det_bbox[1] + 1.0)
        + (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0)
        - inters
    )

    return inters / uni


def compute_metrics(num_true_pos, num_false_pos, num_pos):
    # in numpy, division by zero returns 0; want inf instead:
    precision = num_true_pos / np.maximum(
        num_true_pos + num_false_pos, np.finfo(np.float64).eps
    )
    recall = num_true_pos / num_pos
    return precision, recall, compute_voc_ap(precision, recall)


def compute_voc_ap(prec, recall):
    """Find 'area under PR curve' Monotonic decrease of precision is used
    Ref: https://github.com/Cartucho/mAP and VOC matlab code:

    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Make precision decrease monotonically by finding "local maximas"
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = np.maximum(mpre[index - 1], mpre[index])

    # find all points where recall changes:
    index = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[index + 1] - mrec[index]) * mpre[index + 1])
    return ap


def compute_class_AP(detections, targets, iou_threshold):
    """compute mAP score for a single class' detections

    Args:
        detections (dictionary): dictionary of the form {img_file: {confidence, bbox}}
        targets (dict): dict of the form {img_file: {difficult, det, bbox}}
        iou_threshold (float): threshold for IoU
    """
    # decompose detections -> img_file, confidence, bbox:
    img_files = []
    confidences = []
    bboxes = []

    for img_file, scored_bboxes in detections.items():
        for conf, bbox in scored_bboxes:
            img_files.append(img_file)
            confidences.append(conf)
            bboxes.append(bbox)

    confidences = np.array(confidences)
    bboxes = np.array(bboxes)

    if bboxes.shape[0] == 0:
        return None

    # sort by confidence
    sorted_indices = np.argsort(-confidences)  # sort in descending order
    bboxes = bboxes[sorted_indices, :]
    img_files = [img_files[i] for i in sorted_indices]

    num_true_positive = np.zeros(len(img_files))
    num_false_positive = np.zeros(len(img_files))

    # consume gts with detections
    for index in range(len(img_files)):
        img_file = img_files[index]
        det_bbox = bboxes[index, :]

        gt = targets[img_file]

        iou_scores = np.array([score_iou(det_bbox, gt_bbox) for gt_bbox in gt["bbox"]])
        # sometimes there are actually no gts but there are detections:
        if len(iou_scores) == 0:  # these dets are all fp
            num_false_positive[index] = 1.0
            continue

        max_score, max_index = np.max(iou_scores), np.argmax(iou_scores)
        if max_score > iou_threshold:  # valid detection
            if gt["det"][max_index]:  # gt already consumed by another det
                # mark detection as fp.
                num_false_positive[index] = 1.0  # np array shld be float
                continue
            # consume gt and mark tp.
            gt["det"][max_index] = True
            num_true_positive[index] = 1.0
            continue
        # not a valid detection -> fp
        num_false_positive[index] = 1.0

    # count the number of positives that were available:
    num_positive = 0
    for img_file in detections.keys():
        num_positive += len(targets[img_file]["det"])

    # num of tp/fp is supposed to be a sum:
    num_true_positive = np.cumsum(num_true_positive)
    num_false_positive = np.cumsum(num_false_positive)

    return compute_metrics(num_true_positive, num_false_positive, num_positive)


def compute_mAP(
    detections: dict, gt_filepath: str, iou_threshold: float = 0.5,
):
    """Compute mAP for all classes

    Args:
        detections (dict): dictionary of the form {class: {img_files: {bbox, conf}}}
        gt_filepath (str): ground truth annotation file path.
        iou_threshold (float, optional): IoU threshold used to determine TP detection. Defaults to 0.5.
    """
    # collect image file names from detections:
    classnames = detections.keys()
    class_aps = []
    for classname in classnames:
        # prep dets
        class_detections = detections[classname]
        imagefiles = class_detections.keys()

        # prep gts
        class_gts = load_gt_file(gt_filepath, imagefiles, classnames=classnames)[
            classname
        ]

        _, _, ap = compute_class_AP(
            class_detections, class_gts, iou_threshold=iou_threshold
        )
        class_aps.append(ap)

    return np.mean(np.array(class_aps)), class_aps
