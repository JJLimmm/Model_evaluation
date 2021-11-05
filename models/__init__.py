from .rknn_model import RKNNModel
from .v5_rknn_model import v5RKNNModel


try:
    from .onnx_model import ONNXModel
    from .v5_onnx_model import v5ONNXModel
except ImportError:
    print("ONNXModel not available")
