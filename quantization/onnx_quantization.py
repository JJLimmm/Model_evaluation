from PIL import Image
import numpy as np
import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantType,
)
import os
import argparse


def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.ANTIALIAS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1])  # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (
            image_data[channel, :, :] / 255 - mean[channel]
        ) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data


def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name

        image_data = preprocess_image(image_filepath, height, width)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class YOLOXDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(
                self.image_folder, image_height, image_width, size_limit=0
            )
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter(
                [{input_name: nhwc_data} for nhwc_data in nhwc_data_list]
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
        required=True,
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
    image_height = ort_session.get_inputs()[0].shape[-2]
    image_width = ort_session.get_inputs()[0].shape[-1]
    dr = YOLOXDataReader(args.cal)
    quantize_static(
        model_input,
        model_output,
        dr,
        args.op_types_to_quantize,
        args.per_channel,
        args.reduce_range,
        args.activation_type,
        args.weight_type,
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

