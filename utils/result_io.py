# from ast import parse
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from tqdm import tqdm


def annotation_parser(filename):
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def create_gt_file(filepath, imagefiles, annotations_path):
    annotation_details = {}
    pbar = tqdm(imagefiles, total=len(imagefiles), desc="Parsing annotations")
    for imagefile in pbar:
        annotation_path = Path(os.path.join(annotations_path, imagefile)).with_suffix(
            ".xml"
        )
        pbar.set_description(f"Parsing annotations in {annotation_path}")
        annotation_details[imagefile] = annotation_parser(annotation_path)

    # save gt to pickle file:
    with open(filepath, "wb") as f:  # open/create binary file
        pickle.dump(annotation_details, f)


def load_gt_file(filepath, imagefiles, classnames=None):
    """Load ground truths from a saved pickle file.

    Args:
        filepath (str): path to pickle file.
        imagefiles (iterable): iterable containing names of image files
        classnames (iterable, optional): classnames of dataset. Defaults to None.

    Returns:
        dict[dict[dict]]: Nested dict of form {classnames: {imagefile: {bbox, difficult, det}}}
    """

    with open(filepath, "rb") as f:  # read binary file
        gt = pickle.load(f)

    # find classes:
    if classnames is None:
        classnames = set()
        for imagefile in imagefiles:
            classnames.update([obj["name"] for obj in gt[imagefile]])

    class_gts = {classname: {} for classname in classnames}

    for classname in classnames:
        pbar = tqdm(imagefiles, total=len(imagefiles))
        num_positive = 0  # TODO save it somewhere
        for imagefile in pbar:
            pbar.set_description(f"Parsing gt for {classname} in {imagefile}")
            try:
                class_objects = [
                    obj for obj in gt[imagefile] if obj["name"] == classname
                ]
                bbox = np.array([x["bbox"] for x in class_objects])
                difficult = np.array([x["difficult"] for x in class_objects]).astype(
                    np.bool
                )
                det = [False] * len(class_objects)
                num_positive = num_positive + sum(~difficult)
                class_gts[classname][imagefile] = {
                    "bbox": bbox,
                    "difficult": difficult,
                    "det": det,
                }
            except KeyError:
                print(f"Could not load ground truth of {imagefile}")
                continue
        pbar.close()

    return class_gts


def parse_image_set(txt_file):
    with open(txt_file, "r") as f:
        image_files = f.readlines()
    image_files = [image_file.strip() for image_file in image_files]
    return image_files


if __name__ == "__main__":
    imgfiles = parse_image_set("test_data/ImageSets/main.txt")
    """print(imgfiles[:3])
    create_gt_file(
        "test_data/gt_files/gt.pkl", imgfiles, "test_data/Annotations",
    )"""

    gt_dict = load_gt_file("test_data/gt_files/gt.pkl", imgfiles)
