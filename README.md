# rknn_evaluation

Kit for evaluating rknn models against their onnx models.

## Usage

l. Use [rknn_exporter.py]("./utils/rknn_exporter.py") to convert the onnx model into an rknn model.
l. Use [rknn_demo.py]("./rknn_demo.py") to perform inference and visualization on images included in ./test_data (e.g. `python rknn_demo.py model_name.rknn`)


## Main Changelog

- Fixed inference issues with RKNN model.
- RKNN export script added.
- Now supports environments with and without onnxruntime.


## Known Issues

- The current RKNN model cannot run on Jupyter Notebook when using rknn-toolkit 1.7


## Requirements

- **RKNN Toolkit 1.7 is currently being used for export and will be further used for quantization via onnx.**
- RKNN Toolkit environment setup for 1.6: [bash script to install rknn-toolkit 1.6.1]("./setup/rknn_setup.sh")
- onnxruntime (currently using cpu version): `pip install onnxruntime` or `pip install onnxruntime-gpu`
- Potential conflict between onnxruntime 1.8.1 and rknn-toolkit 1.6.1: rknn-toolkit requires numpy-1.16 but onnxruntime requires numpy-1.19; currently using numpy-1.19, which might cause problems with rknn-toolkit. Further verification  needed.

## Future work

- [X] Experiment with numpy versions to find potential conflicts
- [ ] Support onnx quantization
- [ ] Investigate environment setup to allow simultaneous testing of onnx and rknn models.
- [ ] Include CenterNet processing
- [ ] Evaluation metrics
