from ultralytics import YOLO
from clearml import Task
from utils.utils import read_json

# create an dataset experiment
task = Task.init(project_name="ai_volleyball_scouter", task_name="Ball Detector Pipeline step 2: Train")

# only create the task, we will actually execute it later
task.execute_remotely()

config = read_json("cfg/ball_cfg.json")
dataset_loc = f"{config['main_folder']['project']}-{config["main_folder"]["version"]}"
args = config["train"]["args"]

task.connect(args)

model = YOLO(config["train"]["model"])
model.train(data = f"{dataset_loc}/data.yaml",**args)

