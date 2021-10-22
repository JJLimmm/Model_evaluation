import onnxruntime


class ONNXModel:
    def __init__(self, model_path):
        self.model = onnxruntime.InferenceSession(model_path)
        self._input_name = self.model.get_inputs()[0].name

    def forward(self, image):
        # image need to be expanded into batch
        if len(image.shape) == 3:  # expand dim to 4
            image = image[None, :, :, :]
        output = self.model.run(None, {self._input_name: image})
        return output

    def close(self):
        pass  # Do nothing.
