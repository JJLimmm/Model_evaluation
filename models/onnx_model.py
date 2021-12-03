import onnxruntime


class ONNXModel:
    def __init__(self, model_path, model_type):
        self.model = onnxruntime.InferenceSession(model_path)
        self._input_name = self.model.get_inputs()[0].name
        self.model_type = model_type

    def forward(self, image):
        if self.model_type == "x":
            # image need to be expanded into batch
            if len(image.shape) == 3:  # expand dim to 4
                image = image[None, :, :, :]
            output = self.model.run(None, {self._input_name: image})
            return output
        elif self.model_type == "v5":
            output = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: image})
            return output

    def close(self):
        pass  # Do nothing.
