import numpy as np
from .result_io import load_gt_file, parse_image_set

from typing import Optional


def score_iou(det_bbox, gt_bbox):
    """Compute IoU between a single bboxes

    Args:
        det_bbox (ndarray): detection bbox in xyxy format
        gt_bbox (ndarray): ground truth bbox in xyxy format format

    Returns:
        float: IoU score
    """
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
        targets (dict): dict of the form {img_file: [{difficult, bbox, ...},..]}
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
        detections (dict): dictionary of the form {class: {img_files: {confidence, bbox}}}
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


def plot_confusion_matrix(
    detections,
    gt_filepath,
    classnames,
    conf_threshold=0.25,
    iou_threshold=0.5,
    normalize=True,
):
    num_classes = len(classnames)
    class_indexes = dict(zip(classnames, range(num_classes)))
    matrix = np.zeros((num_classes + 1, num_classes + 1))  # include background

    # filter and convert detections to form {img_id: [cls, bbox]}
    img_detections = {}
    img_gts = {}

    for classname in classnames:
        # prep dets
        class_detections = detections[classname]

        for image_file in class_detections.keys():
            img_detections.setdefault(image_file, [])

            bboxes = list(
                map(
                    lambda x: x[1],
                    filter(
                        lambda x: x[0] > conf_threshold,
                        detections[classname][image_file],
                    ),
                )
            )
            img_detections[image_file].extend([[bbox, classname] for bbox in bboxes])

    # arrange ground truths in a similar manner
    ground_truths = load_gt_file(gt_filepath, img_detections.keys(), classnames)
    for classname in classnames:
        class_gts = ground_truths[classname]  # class_gt is a dictionary with
        for image_file in class_gts.keys():
            img_gts.setdefault(image_file, [])
            img_gts[image_file].extend(
                [[bbox, classname] for bbox in class_gts[image_file]["bbox"]]
            )

    for image_file in img_detections.keys():
        # compare iou of all the bboxes. Each detection will resolve to a ground truth (or background)
        # compute IoU matrix: N_det x N_gt matrix of pairwise bbox IoUs
        gts = img_gts[image_file]
        dets = img_detections[image_file]

        iou_matrix = [
            [score_iou(det_bbox, gt_bbox) for gt_bbox, _ in gts] for det_bbox, _ in dets
        ]

        iou_matrix = np.array(iou_matrix)
        valid_indexes = np.where(iou_matrix > iou_threshold)

        # get matches based on the ious.
        matches = []
        for i in range(valid_indexes[0].shape[0]):
            det_index, gt_index = valid_indexes[0][i], valid_indexes[1][i]
            matches.append([gt_index, det_index, iou_matrix[det_index, gt_index]])
        matches = np.array(matches)

        # print(len(dets))
        # print(len(gts))
        # print(iou_matrix.shape)
        # print(valid_indexes)
        # print(matches.shape)

        if matches.shape[0] > 0:  # valid match exists
            # simple match assignment based on iou scores

            ranked_matches = matches[:, 2].argsort()[::-1]  # best match first
            matches = matches[ranked_matches]

            # np.unique selects the first instance of a unique element encountered;
            # resolve ground truths to their top detection
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            # do the same to resolve detections to their top ground truth
            ranked_matches = matches[:, 2].argsort()[::-1]  # best match first
            matches = matches[ranked_matches]
            matches = matches[np.unique(matches[:, 2], return_index=True)[1]]

            # update confusion matrix based on matches
            gt_indexes, det_indexes, _ = matches.T.astype(
                np.int16
            )  # cast to integer for index to be valid subscriptable type
            det_classes = np.array(dets)[:, 1]

            for index, classname in enumerate(det_classes):
                if not any(det_indexes == index):  # detection could not be matched; FP
                    matrix[class_indexes[classname], num_classes] += 1

            # print(det_classes[det_indexes])
            # print(det_indexes)
            # det_classes[det_indexes]

        for index, (_, classname) in enumerate(gts):
            try:  # TODO restructure this logic
                matched_detections = gt_indexes == index
            except:
                matched_detections = [0]
            if matches.shape[0] > 0 and sum(matched_detections) == 1:
                matrix[
                    list(
                        map(
                            lambda x: class_indexes[x],
                            det_classes[det_indexes[matched_detections]],
                        )
                    ),
                    class_indexes[classname],
                ] += 1
            else:
                matrix[
                    num_classes, class_indexes[classname]
                ] += 1  # no detection for gt; Background FN

    # use confusion matrix to count number of TP, FP, FN:
    # cannot use matrix.diagnonal() in older ver of numpy
    tp = np.diagonal(matrix).copy()
    # fn = matrix.sum(0) - tp
    fp = matrix.sum(1) - tp

    # create plot of confusion matrix:
    try:
        import seaborn as sn
        import matplotlib.pyplot as plt
        import warnings

        array = matrix / (
            (matrix.sum(0).reshape(1, -1) + 1e-6) if normalize else 1
        )  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if num_classes < 50 else 0.8)  # set label size
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(
                array,
                annot=True,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                xticklabels=classnames + ["background FP"],
                yticklabels=classnames + ["background FN"],
            ).set_facecolor((1, 1, 1))
        fig.axes[0].set_title(
            f"Confusion matrix: {round(tp.sum())} TP, {round(fp.sum())} FP + FN"
        )
        fig.axes[0].set_xlabel("True")
        fig.axes[0].set_ylabel("Predicted")
        return fig

    except Exception as e:
        print(f"Confusion matrix plot failed: {e}")
