from PIL import Image
import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
)
import os
import argparse


def preprocess_image(image_path, height, width, swap=[0, 1, 2], channels=3):
    """Normalize image to be used for model inference purposes

    Args:
        image_path (str): path to image file
        height (int): height for resized image
        width (int): width for resized image
        swap (list, optional): swap order of channels by index. Defaults to [0, 1, 2].
        channels (int, optional): Currently redundant. Defaults to 3.

    Returns:
        ndarry: preprocessed image
    """
    image = Image.open(image_path)
    image = image.resize((width, height), Image.ANTIALIAS)
    image_data = np.asarray(image).astype(np.float32)

    # TODO Support both CHW and HWC
    # Swap of [2, 0, 1] converts image in HWC to CHW
    # Swap of [0, 1, 2] (default) maintains image in HWC
    image_data = image_data.transpose(swap)

    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (
            image_data[channel, :, :] / 255 - mean[channel]
        ) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data


def preprocess_func(images_folder, height, width, swap, size_limit=0):
    """ wrapper function to preprocess images into a batch """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name

        image_data = preprocess_image(image_filepath, height, width, swap=swap)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class YOLOXDataReader(CalibrationDataReader):
    """Read and preprocess data for YOLOX model"""

    def __init__(self, calibration_image_folder, input_name, input_shape):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.input_name = input_name
        # self.image_height, self.image_width = input_shape
        self.input_shape = input_shape

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(
                self.image_folder, *self.input_shape, size_limit=0, swap=[2, 0, 1]
            )
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter(
                [{self.input_name: nhwc_data} for nhwc_data in nhwc_data_list]
            )
        return next(self.enum_data_dicts, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Quantization methods")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the input onnx file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.onnx",
        help="Path to the output quantized onnx file",
    )
    parser.add_argument(
        "--cal",
        type=str,
        default="./test_data/Images",
        help="Path to folder containing calibration images",
    )
    parser.add_argument(
        "-ops",
        "--op_types_to_quantize",
        type=list,
        default=[],
        help="Types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default",
    )
    parser.add_argument(
        "-c",
        "--per_channel",
        type=bool,
        default=False,
        help="Quantize weights per channel",
    )
    parser.add_argument(
        "-r",
        "--reduce_range",
        type=bool,
        default=False,
        help="Quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode",
    )
    parser.add_argument(
        "-a",
        "--activation_type",
        type=str,
        default=QuantType.QUInt8,
        help="uint8 or int8",
    )
    parser.add_argument(
        "-w", "--weight_type", type=str, default=QuantType.QUInt8, help="uint8 or int8"
    )
    parser.add_argument(
        "-q",
        "--quant_nodes",
        type=list,
        default=[],
        help="List of nodes names to quantize. When this list is not None only the nodes in this list are quantized.\nexample:['Conv__224','Conv__252']",
    )
    parser.add_argument(
        "-e",
        "--exclude_nodes",
        type=list,
        default=[],
        help="List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.",
    )
    args = parser.parse_args()

    atype = args.activation_type
    if atype == "uint8":
        atype = QuantType.QUInt8
    elif atype == "int8":
        atype = QuantType.QInt8
    elif atype != QuantType.QUInt8:
        raise ValueError("Only uint8 or int8 is supported")
    wtype = args.weight_type
    if wtype == "uint8":
        wtype = QuantType.QUInt8
    elif wtype == "int8":
        wtype = QuantType.QInt8
    elif wtype != QuantType.QUInt8:
        raise ValueError("Only uint8 or int8 is supported")

    model_input = args.input
    model_output = args.output
    ort_session = onnxruntime.InferenceSession(model_input)
    input_name = ort_session.get_inputs()[0].name

    # Consider two possibilities: B x H x W x C or B x C x H x W
    # Assumption: C < H & W
    input_shape = ort_session.get_inputs()[0].shape
    if input_shape[-1] > input_shape[-3]:
        # last index is W; format is B x C x H x W
        image_height = input_shape[-2]
        image_width = input_shape[-1]
    else:
        # Format is B x H x W x C
        image_height = input_shape[-3]
        image_width = input_shape[-2]

    dr = YOLOXDataReader(args.cal, input_name, (image_height, image_width))
    quantize_static(
        model_input,
        model_output,
        dr,
        args.op_types_to_quantize,
        args.per_channel,
        args.reduce_range,
        atype,
        wtype,
        args.quant_nodes,
        args.exclude_nodes,
    )
    print(
        "ONNX full precision model size (MB):",
        os.path.getsize(model_input) / (1024 * 1024),
    )
    print(
        "ONNX quantized model size (MB):", os.path.getsize(model_output) / (1024 * 1024)
    )
