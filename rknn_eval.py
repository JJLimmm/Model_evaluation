from utils.result_io import parse_image_set
from models import RKNNModel

try:
    from models import ONNXModel
except ImportError:
    print("ONNXModel not available in this environment")

from yolox_processing import postprocess, preprocess, decompose_detections
from utils import compute_mAP, create_gt_file, plot_confusion_matrix

import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path

import time
from tqdm import tqdm

import argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--model_type", type=str, default="rknn")
    parser.add_argument("--onnx", action="store_true", help="run demo with onnx model")
    parser.add_argument("--use_sim", type=bool, default=False)
    parser.add_argument("--dev", type=str, default="TM018083200400463")
    parser.add_argument(
        "--res",
        nargs="+",
        type=int,
        default=[512, 512],
        help="Input resolution. If only 1 argument is provided, it is broadcast to 2 dimensions",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy preprocessing for yolox models",
    )
    parser.add_argument(
        "--names",
        required=True,
        nargs="+",
        type=str,
        help="Class names (MUST be in indexed order)",
    )
    parser.add_argument(
        "--absconfusion",
        action="store_false",
        help="Do not normalize confusion matrix; use absolute values.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="name of dataset folder in ./test_data",
    )

    return parser


def preprocess_image_batch(
    image_file_batch, resize_shape=(512, 512), dtype=np.float32, legacy=False
):
    """Prepare a batch of images for inference

    Args:
        image_file_batch (list[ndarray]): list of images that form a batch
        resize_shape (tuple, optional): shape for preprocessed images. Defaults to (512, 512).
        dtype (np.dtype, optional): final data type for preprocessed images. Defaults to np.float16.

    Returns:
        list, list: list of images and list of dictionaries containing metadata.
    """
    images = []
    meta_data = []

    for image_file in image_file_batch:
        file_name = image_file.with_suffix("").parts[-1]
        image = cv2.imread(str(image_file))

        raw_image = image.copy()
        image, ratio = preprocess(image, resize_shape, legacy=legacy)
        image = np.expand_dims(image, axis=0)  # further rknn processing
        image = np.array(image, dtype=dtype)

        images.append(image)
        meta_data.append(
            {
                "raw_img": raw_image,
                "ratio": ratio,
                "test_size": resize_shape,
                "image_file": file_name,
            }
        )

    return images, meta_data


def process_outputs(outputs, meta_data, dets_dict, class_names):
    """Convert a batch of model outputs into detections.
    Mutates input detection dictionary to store the output detections.

    Args:
        outputs (list[ndarray]): list of raw model outputs
        meta_data (list[dict]): list of meta data for each output/image
        dets_dict (dict): dictionary of existing detections
        class_names (list): list to map detection indexes to class names

    Returns:
        dict: mutated dets_dict that contains new detections
    """
    # post process the outputs:
    for output, data in zip(outputs, meta_data):

        # print(data)
        dets = postprocess(output, data, create_visualization=False)
        image_file = data["image_file"]

        if dets is None:  # no detections to add
            continue

        # format det and add to dictionary
        # dictionary has the form {class: {img_file: [[conf, bbox]...]}}
        class_detections = {x: [] for x in class_names}
        # sort into classes:
        for bbox, score, class_index in zip(*decompose_detections(dets)):
            class_detections[class_names[int(class_index)]].append([score, bbox])
        # merge with dets_dict:
        for class_name in class_detections.keys():
            dets_dict[class_name][image_file] = class_detections[class_name]
    return dets_dict


def build_gt_file(pkl_filename, image_folder, annotations_folder, image_set_file=None):
    """Construct collated ground truth file from evaluation set.

    Args:
        pkl_filename (str): path to write ground truth pickle (pkl) file
        image_folder (str): path to ground truth images. Used to get image file names if image_set_file is None
        annotations_folder (str): path to root folder for ground truth annotations
        image_set_file (str, optional): path to image set text file. If provided, is used in place of file names in image folder. Defaults to None.
    """

    try:
        # use the image set file for image file names if possible
        image_files = parse_image_set(image_set_file)

    except:
        image_files = []
        for image_file in Path(image_folder).iterdir():
            image_files.append(image_file.with_suffix("").parts[-1])

    create_gt_file(pkl_filename, image_files, annotations_folder)


def evaluate(
    model,
    resize_shape=(512, 512),
    gt_filename="./test_data/gt_files/gt.pkl",
    images_path="./test_data/Images",
    annotations_path="./test_data/Annotations",
    image_set_file="./test_data/ImageSets/main.txt",
    batch_size=16,
    result_path="./test_results",
    class_names=["A", "B"],
    legacy=False,
    normalize_confusion_matrix=True,
):
    """Evaluate a model on VOC mAP

    Args:
        rknn_model (model): provides inference
        input_size (list/tuple): image input size in H x W.
        gt_filename (str, optional): ground truth pickle file; will be created at location if does not exist. Defaults to "./test_data/gt_files/gt.pkl".
        images_path (str, optional): path to image folder. Defaults to "./test_data/Images".
        annotations_path (str, optional): path to annotations folder. Defaults to "./test_data/Annotations".
        image_set_file (str, optional): path to imageset txt file. Defaults to "./test_data/ImageSets/main.txt".
        batch_size (int, optional): affects number of images stored in memory. Defaults to 32.
        result_path (str, optional): Currently redundant. Defaults to "./test_results".
        class_names (list, optional): class names used in groundtruth. Must match else gt won't load. Defaults to ["A", "B"].
    """
    assert len(resize_shape) == 1 or len(resize_shape) == 2
    if len(resize_shape) == 1:
        resize_shape = (resize_shape[0], resize_shape[0])
    # check if gt file already exists:
    if not os.path.exists(gt_filename):
        # need to build new gt file
        build_gt_file(gt_filename, images_path, annotations_path, image_set_file)

    # create folder for test results
    if result_path == "./test_results":
        result_path = os.path.join(result_path, str(datetime.now()))
    Path(result_path).mkdir(exist_ok=True, parents=True)

    # perform evaluation in batches.
    # non-batch variables:
    inference_times = []
    detections = {x: {} for x in class_names}

    # batch variables:
    image_file_batch = []

    for image_file in tqdm(
        Path(images_path).iterdir(),
        desc="Collecting detections",
        total=len(os.listdir(images_path)),
    ):
        # accumulate batch of files:
        if len(image_file_batch) < batch_size:
            image_file_batch.append(image_file)
            continue

        # batch process:
        images, meta_data = preprocess_image_batch(
            image_file_batch, resize_shape=resize_shape, legacy=legacy
        )
        image_file_batch = []
        outputs = []

        # perform inference
        for image in images:
            start_time = time.time()
            output = model.forward(image)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            outputs.append(output)

        # process and save the outputs for evaluation
        detections = process_outputs(outputs, meta_data, detections, class_names)

    # evaluate detections obtained:
    mAP, class_aps = compute_mAP(detections, gt_filename)

    model.close()
    print(f"Average Inference time: {np.mean(np.array(inference_times))}")
    print(f"Class aps: {class_aps}")
    print(f"Final mAP: {mAP}")

    confusion_matrix = plot_confusion_matrix(
        detections, gt_filename, class_names, normalize=normalize_confusion_matrix
    )
    confusion_matrix.savefig(os.path.join(result_path, "confusion_matrix.png"), dpi=250)


if __name__ == "__main__":
    try:
        from models import ONNXModel
    except ImportError:
        print(
            "ONNXModel not available in this environment. Need to install onnxruntime."
        )

    args = make_parser().parse_args()

    if args.model_type == "onnx" or args.onnx:
        model = ONNXModel(os.path.join("./rknn_exports", args.model_path))
    else:
        model = RKNNModel(
            os.path.join("./rknn_exports", args.model_path),
            use_sim=args.use_sim,
            device_id=args.dev,
        )

    if args.dataset is None:
        evaluate(
            model,
            class_names=args.names,
            resize_shape=args.res,
            legacy=False,  # args.legacy,
            normalize_confusion_matrix=args.absconfusion,
        )
    else:
        evaluate(
            model,
            class_names=args.names,
            resize_shape=args.res,
            legacy=False,  # args.legacy,
            gt_filename=f"./test_data/gt_files/{args.dataset}.pkl",
            images_path=f"./test_data/{args.dataset}/Images",
            annotations_path=f"./test_data/{args.dataset}/Annotations",
            image_set_file=f"./test_data/{args.dataset}/ImageSets/main.txt",
            normalize_confusion_matrix=args.absconfusion,
        )

    model.close()
