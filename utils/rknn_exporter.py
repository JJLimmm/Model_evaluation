from rknn.api import RKNN
import onnx
from onnx import shape_inference
from typing import Optional
import os

LOG_FOLDER = "logs"
QUANT_KEYS = [
    "mean_values",
    "std_values",
    "batch_size",
    "epochs",
    "dtype",
    "algo",
    "dataset",
]


def onnx_to_rknn(
    onnx_model_path,
    input_size: list,
    rknn_model_name: str = "rknn_model",
    rknn_export_dir: str = "./rknn_exports",
    log_onnx_validation=False,
    verbose=True,
    perform_quant=False,
    quant_info: Optional[dict] = None,
):
    onnx_model = onnx.load(onnx_model_path)

    # perform onnx model validation:
    # onnx.checker.check_model(onnx_model)
    if log_onnx_validation:
        shape_inference_model = shape_inference.infer_shapes(onnx_model)
        shape_info_file = open(
            os.path.join(
                rknn_export_dir,
                LOG_FOLDER,
                f"{rknn_model_name}_onnx_shape_inference.txt",
            ),
            "w",
        )
        shape_info_file.write(f"Graph:\n{shape_inference_model.graph.value_info}")
        shape_info_file.close()

        onnx_graph_file = open(
            os.path.join(
                rknn_export_dir, LOG_FOLDER, f"{rknn_model_name}_onnx_graph.txt"
            ),
            "w",
        )
        onnx_graph_file.write(onnx.helper.printable_graph(onnx_model.graph))
        onnx_graph_file.close()

    print(f"onnx model {onnx_model_path} has been verified.")

    # Verify model export params first
    if perform_quant:
        assert (
            quant_info is not None
        ), "quant_info argument required to perform quantization"
        print(f"Quantization will be performed with the following parameters:")
        for key in QUANT_KEYS:
            print(f"{key} :  {quant_info[key]}")
    assert len(input_size) == 3, "Expected input_size list to be length 3"

    rknn = RKNN(verbose=verbose)

    if perform_quant:
        rknn.config(
            target_platform=["rk1808"],
            optimization_level=3,
            output_optimize=1,
            mean_values=[quant_info["mean_values"]],
            std_values=[quant_info["std_values"]],
            reorder_channel="0 1 2",  # TODO
            quantized_algorithm=quant_info["algo"],
            quantized_dtype=quant_info["dtype"],
            batch_size=quant_info["batch_size"],
            epochs=quant_info["epochs"],
        )
    else:
        rknn.config(
            target_platform=["rk1808"],
            optimization_level=3,
            output_optimize=1,
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
            reorder_channel="0 1 2",  # TODO
        )

    if rknn.load_onnx(
        model=onnx_model_path,
        inputs=["images"],
        input_size_list=[input_size],
        # outputs=["output0", "output1", "output2"],
    ):
        print("Failed to import onnx model into RKNN")
        return False

    if perform_quant:
        build_status = rknn.build(
            do_quantization=perform_quant, dataset=quant_info["dataset"]
        )
    else:
        build_status = rknn.build(do_quantization=perform_quant)

    export_status = rknn.export_rknn(
        os.path.join(rknn_export_dir, f"{rknn_model_name}.rknn")
    )

    print(f"Model export process completed with status: {export_status}")
    return not (bool(export_status) or bool(build_status))


if __name__ == "__main__":
    model_path = "./rknn_exports/onnx/final0.onnx"

    try:
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(model_path)
        input_shape = ort_session.get_inputs()[0].shape
        if len(input_shape) == 4:
            input_shape = input_shape[1:]  # drop batch size
    except ImportError:
        print(
            f"onnxruntime module not available. Unable to infer input shape; falling back on default input shape"
        )
        input_shape = [3, 512, 512]

    print(input_shape)

    quant_info = {
        "mean_values": [123.7, 116.3, 103.5],
        "std_values": [58.4, 57.1, 57.4],
        "batch_size": 8,
        "epochs": 50,
        "dtype": "dynamic_fixed_point-i16",
        "algo": "normal",
        "dataset": "../data/helmet_quantization/Images/helmet_quantization_data.txt",
    }

    onnx_to_rknn(
        model_path,
        input_shape,
        rknn_model_name="final0QAT",
        log_onnx_validation=True,
        perform_quant=False,
        quant_info=quant_info,
    )
