from numpy.core.fromnumeric import resize
from models import RKNNModel, ONNXModel
from yolov5_processing import *

import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import time
import argparse

IMAGE_EXT = (".jpg", ".jpeg", ".webp", ".bmp", ".png")  # must be tuple
device_type="rk1808",

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("model_type",type = str, default="v5",help="v5 or x")
    parser.add_argument("--use_sim", type=bool, default=False)
    parser.add_argument(
        "--in_res",
        nargs="+",
        type=int,
        default=[512, 512],
        help="Input only 1 resolution.",
    )
    parser.add_argument("--test_dir",type=str,default="./test_data/Images")
    parser.add_argument("--test_save",type=str,default="./test_results")
    parser.add_argument("--dev", type=str, default="TM018083200400463")

    return parser


def rknn_post_process(outputs,img_size):
    # full post process 
    input0_data = outputs[1].transpose(0,1,4,2,3)
    input1_data = outputs[2].transpose(0,1,4,2,3)
    input2_data = outputs[3].transpose(0,1,4,2,3)

    input0_data = input0_data.reshape(*input0_data.shape[1:])
    input1_data = input1_data.reshape(*input1_data.shape[1:])
    input2_data = input2_data.reshape(*input2_data.shape[1:])

    input_data = []
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    boxes, classes, scores = rknn_yolov5_post_process_full(input_data,img_size)
    return boxes,classes,scores

def rknn_vis_save(img,boxes,scores,classes,output_folder,img_name):
    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        rknn_draw(img_1, boxes, scores, classes)
    print("Saving result to",output_folder+'/'+img_name)
    if not cv2.imwrite(output_folder+'/'+img_name.split('/')[-1], img_1):
        raise Exception("Could not write image")

def onnx_vis_save(img,im0s,pred,output_folder,img_name):
    for det in pred:
        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=1, example=str(CLASSES))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{CLASSES[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
        if not cv2.imwrite(output_folder+'/'+img_name.split('/')[-1], im0):
            raise Exception("Could not write image")

def demo(model,model_type,test_dir,test_save,resize_shape=512):
    output_folder = test_save+'/YOLOv5_'+str(datetime.now())
    print("Inference results will be saved to:",output_folder)
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    if model_type == "onnx":
        dataset = LoadImages(test_dir, img_size=resize_shape[0], stride=64, auto=False)
        for path, img, im0s, vid_cap in dataset:
            img = onnx_preprocess(img)
            start_time = time.time()
            pred = model.forward(img)
            print(f"Infer time: {round(time.time() - start_time, 4)}s")
            pred = torch.tensor(pred)
            pred = onnx_non_max_suppression(pred, 0.25, 0.45)
            onnx_vis_save(img,im0s,pred,output_folder,path.split('/')[-1])
        model.close()
    else:
        for img in Path(test_dir).iterdir():
            img_name = str(img)
            print("Performing inference on:",img_name)
            # ignore non-image files:
            if not img_name.endswith(IMAGE_EXT):
                continue

            img = cv2.imread(str(img))
            img, ratio, (dw, dh) = rknn_letterbox(img, new_shape=(*resize_shape,*resize_shape))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Inference
            print('--> Running model')
            start_time = time.time()
            outputs = model.forward(img)
            print(f"Infer time: {round(time.time() - start_time, 4)}s")
            boxes,classes,scores = rknn_post_process(outputs,*resize_shape)
            rknn_vis_save(img,boxes,scores,classes,output_folder,img_name)

        

    


if __name__ == '__main__':
    CLASSES = ("head","helmet")
    colors = Colors()
    
    args = make_parser().parse_args()
    # Create RKNN object
    engine = args.model_path.split('.')[-1]
    # pre-process config
    if engine == "rknn":
        model = RKNNModel(args.model_path,args.model_type,args.use_sim,device_id = args.dev)

    elif engine == "onnx":
        if not os.path.exists(args.model_path):
            print('Model does not exist')
            exit(-1)
        model = ONNXModel(args.model_path,args.model_type)
        
    demo(model,engine,args.test_dir,args.test_save,args.in_res)