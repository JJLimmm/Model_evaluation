from models import RKNNModel
from yolox_processing import postprocess, preprocess

import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path

import time

import argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--use_sim", type=bool, default=False)

    return parser


def preprocess_image_folder(path, resize_shape=(512, 512), dtype=np.float16):
    images = []
    meta_data = []
    for image_file in Path(path).iterdir():
        image = cv2.imread(str(image_file))

        raw_image = image.copy()
        image, ratio = preprocess(image, resize_shape)
        image = np.expand_dims(image, axis=0)  # further rknn processing
        image = np.array(image, dtype=dtype)

        images.append(image)
        meta_data.append(
            {"raw_img": raw_image, "ratio": ratio, "test_size": resize_shape}
        )

    return images, meta_data


def vis_batch(outputs, meta_data, path="./test_results"):
    if path == "./test_results":
        output_folder = os.path.join(path, str(datetime.now()))
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    for index, (output, data) in enumerate(zip(outputs, meta_data)):
        vis_image = postprocess(output, data)
        cv2.imwrite(str(os.path.join(output_folder, f"vis{index}.jpg")), vis_image)


def demo(rknn_model, images_path="./test_data"):
    images, meta_data = preprocess_image_folder(images_path)
    outputs = []
    for image in images:
        start_time = time.time()
        output = rknn_model.forward(image)
        print(f"Infer time: {round(time.time() - start_time, 4)}s")
        outputs.append(output)
    rknn_model.close()

    vis_batch(outputs, meta_data)


if __name__ == "__main__":
    args = make_parser().parse_args()
    # model = RKNNModel("./rknn_exports/yolox_tiny_qt.rknn", use_sim=True)
    model = RKNNModel(
        os.path.join("./rknn_exports", args.model_path), use_sim=args.use_sim
    )
    demo(model)
