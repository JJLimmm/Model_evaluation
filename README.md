# rknn_evaluation

Kit for evaluating rknn models against their onnx models.

## Usage

1. Use [rknn_exporter.py](utils/rknn_exporter.py) to convert the onnx model into an rknn model.
1. Use [rknn_demo.py](rknn_demo.py) to perform inference and visualization on images included in ./test_data (e.g. `python rknn_demo.py model_name.rknn --use_sim True`)

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
┃┗ 📜rknn_exporter.py
┣📂setup
┃┗ 📜rknn_setup.sh
┣ 📂yolox_processing
┃ ┣ 📜processing.py
┃ ┗ 📜visualization.py
┣📂test_data
┃┣ 📂Annotations
┃┣ 📂Images
┃┣ 📂gt_files
┣📂rknn_exports
┃┣ 📂example
┃┃ ┣ 📜yolox_tiny_v3.onnx
┃┃ ┗ 📜yolox_tiny_v3.rknn
 ┣ 📂logs
┣ 📜rknn_demo.py
┗ 📜rknn_eval.py
```

Overall, primary scripts are kept at the root of the project, while all other auxillary files are kept in processing, models, or utils respectively.

## Main Changelog

- Fixed inference issues with RKNN model.
- RKNN export script added.
- Now supports environments with and without onnxruntime.

## Known Issues

- The current RKNN model cannot run on Jupyter Notebook when using rknn-toolkit 1.7

## Requirements

- **RKNN Toolkit 1.7 is currently being used for export and will be further used for quantization via onnx.**
- RKNN Toolkit environment setup for 1.6: [bash script to install rknn-toolkit 1.6.1](/setup/rknn_setup.sh)
- onnxruntime (currently using cpu version): `pip install onnxruntime` or `pip install onnxruntime-gpu`
- Potential conflict between onnxruntime 1.8.1 and rknn-toolkit 1.6.1: rknn-toolkit requires numpy-1.16 but onnxruntime requires numpy-1.19; currently using numpy-1.19, which might cause problems with rknn-toolkit. Further verification  needed.

## Future work

- [X] Experiment with numpy versions to find potential conflicts
- [ ] Support onnx quantization
- [ ] Investigate environment setup to allow simultaneous testing of onnx and rknn models.
- [ ] Include CenterNet processing
- [ ] Evaluation metrics
