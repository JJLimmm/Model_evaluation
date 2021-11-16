from numpy.core.fromnumeric import resize
from models import RKNNModel
from yolox_processing import postprocess, preprocess

import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path

import time

import argparse


IMAGE_EXT = (".jpg", ".jpeg", ".webp", ".bmp", ".png")  # must be tuple


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--img_path", default="./test_data", type=str)
    parser.add_argument("--use_sim", type=bool, default=False)
    parser.add_argument(
        "--res",
        nargs="+",
        type=int,
        default=[512, 512],
        help="Input resolution. If only 1 argument is provided, it is broadcast to 2 dimensions",
    )
    parser.add_argument("--dev", type=str, default="TM018083200400463")
    parser.add_argument("--onnx", action="store_true", help="run demo with onnx model")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy preprocessing for yolox models",
    )

    return parser


def preprocess_image_folder(
    path, resize_shape=(512, 512), dtype=np.float16, legacy=False
):
    images = []
    meta_data = []

    for image_file in Path(path).iterdir():
        # ignore non-image files:
        if not str(image_file).endswith(IMAGE_EXT):
            continue

        image = cv2.imread(str(image_file))

        raw_image = image.copy()
        image, ratio = preprocess(image, resize_shape, legacy=legacy)
        image = np.expand_dims(image, axis=0)  # further rknn processing
        image = np.array(image, dtype=dtype)
        # TODO allow change in dtype for onnx models.

        images.append(image)
        meta_data.append(
            {"raw_img": raw_image, "ratio": ratio, "test_size": resize_shape}
        )

    return images, meta_data


def vis_batch(outputs, meta_data, output_folder="./test_results"):
    if output_folder == "./test_results":
        output_folder = os.path.join(output_folder, str(datetime.now()))
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    for index, (output, data) in enumerate(zip(outputs, meta_data)):
        vis_image = postprocess(output, data, class_names=[str(x) for x in range(80)])
        cv2.imwrite(str(os.path.join(output_folder, f"vis{index}.jpg")), vis_image)


def demo(
    model,
    images_path="./test_data",
    resize_shape=(512, 512),
    dtype=np.float16,
    legacy=False,
):
    assert len(resize_shape) == 1 or len(resize_shape) == 2
    if len(resize_shape) == 1:
        resize_shape = (resize_shape[0], resize_shape[0])
    images, meta_data = preprocess_image_folder(
        images_path, resize_shape=resize_shape, dtype=dtype, legacy=legacy
    )
    outputs = []
    for image in images:
        start_time = time.time()
        output = model.forward(image)
        print(f"Infer time: {round(time.time() - start_time, 4)}s")
        outputs.append(output)
    model.close()

    vis_batch(outputs, meta_data)


if __name__ == "__main__":
    try:
        from models import ONNXModel
    except ImportError:
        print(
            "ONNXModel not available in this environment. Need to install onnxruntime."
        )

    args = make_parser().parse_args()

    if args.onnx:
        model = ONNXModel(os.path.join("./rknn_exports", args.model_path))
        dtype = np.float32
    else:
        model = RKNNModel(
            os.path.join("./rknn_exports", args.model_path),
            use_sim=args.use_sim,
            device_id=args.dev,
        )
        dtype = np.float16
    demo(
        model,
        images_path=args.img_path,
        resize_shape=args.res,
        dtype=dtype,
        legacy=args.legacy,
    )
