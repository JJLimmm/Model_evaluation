from .rknn_model import RKNNModel

try:
    from .onnx_model import ONNXModel
except ImportError:
    print("ONNXModel not available")
