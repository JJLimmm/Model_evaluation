import onnxruntime
import numpy as np

class v5ONNXModel:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
        self._input_name = self.model.get_inputs()[0].name

    def forward(self, image):
        output = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: image})
        return output

    def close(self):
        pass  # Do nothing.
