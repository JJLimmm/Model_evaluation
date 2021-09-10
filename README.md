# rknn_evaluation

Kit for evaluating rknn models against their onnx models.

## Requirements

- RKNN Toolkit environment setup: [bash script to install rknn-toolkit 1.6.1]("./setup/rknn_setup.sh")
- onnxruntime (currently using cpu version): `pip install onnxruntime` or `pip install onnxruntime-gpu`
- matplotlib for visualization `conda install matplotlib`
- Potential conflict between onnxruntime 1.8.1 and rknn-toolkit 1.6.1: rknn-toolkit requires numpy-1.16 but onnxruntime requires numpy-1.19; currently using numpy-1.19, which might cause problems with rknn-toolkit. Further verification  needed.

## Future work

- [ ] Experiment with numpy versions to find potential conflicts
- [ ] Include CenterNet processing
- [ ] Evaluation metrics
