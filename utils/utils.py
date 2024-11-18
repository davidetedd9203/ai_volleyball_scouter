from urllib.request import urlopen
import json
import glob
import shutil
import os

import numpy as np
import cv2
from roboflow import Roboflow

def opencv_urlopen(url: str) -> np.array:
    """
    opens an image from the internet with opencv

    Args:
        url (str): the url of the image

    Returns:
        np.array: teh image as np.array
    """
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def read_json(path: str) -> dict:
    """
    load a json file into a dictionary

    Args:
        path (str): the path of the json file

    Returns:
        dict: dictionary with the json data
    """
    with open(path, "rb") as js:
        cfg = json.load(js)
    return cfg

def get_from_roboflow(rf: Roboflow, workspace: str, project: str, version: int, yolo_version: str, location: str = "data/") -> None:
    """download data from roboflow

    Args:
        rf (Roboflow): the roboflow client
        workspace (str): workspace of roboflow
        project (str): project of roboflow
        version (int): dataset version
        yolo_version (str): yolo version
        location (str): final location of the dataset
    """
    project = rf.workspace(workspace).project(project)
    version = project.version(version)
    dataset = version.download(yolo_version, location = location)
    return dataset


def merge_dataset(start_folder: str, dest_folder: str) -> None:
    """merge two roboflow dataset

    Args:
        start_folder (str): _description_
        dest_folder (str): _description_
    """
    mods = ["train","test","valid"]
    types = ["images", "labels"]
    for mod in mods:
        for typ in types:
            start = f"{start_folder}/{mod}/{typ}"
            dest = f"{dest_folder}/{mod}/{typ}"
            print("BEFORE:")
            print(f"len{start}: {len(glob.glob(f'{start}/*'))}")
            for file_path in glob.glob(f"{start}/*"):
                # Costruisce il percorso di destinazione
                file_name = os.path.basename(file_path)
                destination_path = os.path.join(dest, file_name)

                # Sposta il file
                shutil.move(file_path, destination_path)
            print("THEN:")
            print(f"len{start}: {len(glob.glob(f'{start}/*'))}")
            print(f"len{dest}: {len(glob.glob(f'{dest}/*'))}")