import onnxruntime


class ONNXModel:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
        self._input_name = self.model.get_inputs()[0].name

    def forward(self, image):
        # image need to be expanded into batch
        output = self.model.run(None, {self._input_name: image[None, :, :, :]})
        return output
