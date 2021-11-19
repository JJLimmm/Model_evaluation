from rknn.api import RKNN
import numpy as np


class RKNNModel:
    def __init__(
        self,
        model_path,
        model_type,
        use_sim=True,
        device_type="rk1808",
        device_id="TM018084210400233",
        verbose=False,
        
    ):
        self.model = RKNN(verbose=verbose)
        self.model.load_rknn(model_path)
        self.model_type = model_type
        if use_sim:
            print(f"Using simulated {device_type}")
            self.model.init_runtime()
        else:
            try:
                print(f"Initializing with {device_type}: connecting to {device_id}")
                self.model.init_runtime(target=device_type, device_id=device_id)
            except:
                print(
                    "Could not initialize rknn runtime on actual device; falling back on simulation"
                )
                print("Verify that the device id and type are correct:")
                print(f" {device_id} : {device_type}")
                self.model.init_runtime(target=device_type)

    def forward(self, image):  # TODO input types. docstrings.
        # TODO use image type as reference for model type:
        if self.model_type == "x":
            result = self.model.inference(inputs=[image],
                                        data_format="nchw",
                                        data_type=str(image.dtype),
                                        inputs_pass_through=[1],
                                        )
            return np.array(result)
        elif self.model_type == "v5":
            result = self.model.inference(inputs=[image])
            return result


    def close(self):
        self.model.release()

