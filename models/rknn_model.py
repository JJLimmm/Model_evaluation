from rknn.api import RKNN


class RKNNModel:
    def __init__(
        self,
        model_path,
        use_sim=True,
        device_type="rk1808",
        device_id="TM018084210400233",
    ):
        self.model = RKNN()
        self.model.load_rknn(model_path)
        if use_sim:
            self.model.init_runtime(target=device_type)
        else:
            try:
                self.model.init_runtime(target=device_type, device_id=device_id)
            except:
                print(
                    "Could not initialize rknn runtime on actual device; falling back on simulation"
                )
                print("Verify that the device id and type are correct:")
                print(f" {device_id} : {device_type}")
                self.model.init_runtime(target=device_type)

    def forward(self, image):
        return self.model.inference(inputs=[image])


if __name__ == "__main__":
    import cv2
    import numpy as np

    rknn = RKNNModel("./rknn_exports/yolox_helmet_v2_qt.rknn", use_sim=False)
    # preprocess img:
    img = cv2.imread("./test_data/mpe_phone_07_000185.jpg")

    output = rknn.forward(img)

