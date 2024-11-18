from tqdm.auto import tqdm
import shutil
from clearml import Task, StorageManager
from roboflow import Roboflow
from utils.utils import (
    read_json,
    get_from_roboflow,
    merge_dataset
)

# create an dataset experiment
task = Task.init(project_name="ai_volleyball_scouter", task_name="Ball Detector Pipeline step 1 dataset creation")

# only create the task, we will actually execute it later
task.execute_remotely()

config = read_json("cfg/ball_cfg.json")
api_key = read_json("cfg/roboflow.json")["api_key"]
rf = Roboflow(api_key)

dest_dataset = get_from_roboflow(rf, **config["main_folder"])

for folder in tqdm(config["folders"]):
    dataset = get_from_roboflow(rf, **folder)
    merge_dataset(dest_dataset.location, dataset.location)
    shutil.rmtree(dataset.location)
    
task.upload_artifact('dataset location', artifact_object=dest_dataset.location)


