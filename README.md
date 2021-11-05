# rknn_evaluation

Kit for evaluating rknn models against their onnx models.

## Usage

1. (Optional) Use [onnx_quantization.py](quantization/onnx_quantization.py) to quantize the onnx model.
1. Use [rknn_exporter.py](utils/rknn_exporter.py) to convert the onnx model into an rknn model.
1. YOLOX: Use [rknn_demo_x.py](rknn_demo_x.py) to perform inference and visualization
e.g. `python rknn_demo_x.py model_path.rknn --use_sim True`
1. YOLOv5: Use [rknn_demo_v5.py](rknn_demo_v5.py) to perform inference and visualization. Note that ONNXRuntime works only for non quantized YOLOv5 models.
e.g. 
`python rknn_demo_v5.py model_path.onnx --use_sim True --in_res 640 --model_type onnx --test_dir ./test_data/Images --test_save ./test_results`
`python rknn_demo_v5.py model_path.rknn --use_sim True --in_res 640 --model_type rknn --test_dir ./test_data/Images --test_save ./test_results`
`python rknn_demo_v5.py -h` for details

### ONNX Quantization

Quantization via ONNX is currently supported for YOLOX models and is for now the only method to obtain viable int8/uint8 models.
Example usage:

```bash
python utils/onnx_quantization.py -i rknn_exports/example/yolox_tiny_v3.onnx -o rknn_exports/yolox_tiny_v3_uint8.onnx --cal test_data/Images # only -i is a required flag.

```

### RKNN Model API

The RKNN model has the following workflow and functions:

```python

# initialize model:
model = RKNNModel(model_path, use_sim=False, device_id="", verbose=False)
# device id can be obtained by the cli command: adb devices
# device id will need to be supplied if not using simulation (which is default)

# make inference with preprocessed img:
output = model.forward(preprocessed_image)

# release the RKNN api; currently does not help with RKNN restart problem
model.close() 

```

### Project File Structure

An overview of the project file structure. Only essential files are shown:

```tree
📦rknn_evaluation
┣📂models
┃┣ 📜onnx_model.py
┃┗ 📜rknn_model.py
┣📂utils
┃┣ 📜eval.py
┃┣ 📜result_io.py
┃┣ 📜rknn_exporter.py
┃┗ 📜onnx_quantization.py
┣📂setup
┃┗ 📜rknn_setup.sh
┣📂yolox_processing
┃┣ 📜processing.py
┃┗ 📜visualization.py (not yet addded)
┣📂test_data
┃┣ 📂Annotations
┃┣ 📂Images
┃┣ 📂gt_files
┣📂rknn_exports
┃┣ 📂example
┃┃ ┣ 📜yolox_tiny_v3.onnx
┃┃ ┗ 📜yolox_tiny_v3.rknn
┃┣ 📂logs
┣ 📜rknn_demo.py
┗ 📜rknn_eval.py
```

Overall, primary scripts are kept at the root of the project, while all other auxillary files are kept in processing, models, or utils respectively.

## Main Changelog
- Added YOLOv5 ONNXRuntime and RKNN demo
- Fixed inference issues with RKNN model.
- RKNN export script added.
- Now supports environments with and without onnxruntime.
- Added onnx quantization.
- Added simple rknn model evaluation.

## Known Issues

- The current RKNN model cannot run on Jupyter Notebook when using rknn-toolkit 1.7
- When running the rknn on the chip, the process does not terminate despite `rknn.release()` being called. Process needs to be killed instead.

## Requirements

- **RKNN Toolkit 1.7 is currently being used for export and will be further used for quantization via onnx.**
- RKNN Toolkit environment setup for 1.7: [bash script to install rknn-toolkit 1.7.1](/setup/rknn_setup.sh)
- onnxruntime (currently using cpu version): `pip install onnxruntime` or `pip install onnxruntime-gpu`
- Potential conflict between onnxruntime 1.5.2 and rknn-toolkit 1.7.1: rknn-toolkit requires numpy-1.16 but onnxruntime requires numpy-1.19; currently using numpy-1.19, which might cause problems with rknn-toolkit. No issues thus far.

## Future work

- [X] Experiment with numpy versions to find potential conflicts
- [X] Support onnx quantization
- [X] Investigate environment setup to allow simultaneous testing of onnx and rknn models.
- [ ] Include CenterNet processing
- [X] Include YOLOv5 processing
- [X] Evaluation metrics
