from clearml import PipelineController

pipe = PipelineController(
     name='ball detector train pipeline',
     project='ai_volleyball_scouter',
     version='0.0.1',
     add_pipeline_tags=False,
)
pipe.set_default_execution_queue('default')

pipe.add_step(
    name="dataset_creation",
    base_task_project="ai_volleyball_scouter",
    base_task_name="Ball Detector Pipeline step 1 dataset creation",
)

pipe.add_step(
    name="train_model",
    parents=["dataset_creation"],
    base_task_project="examples",
    base_task_name="Ball Detector Pipeline step 2: Train",
)

pipe.execute_remotely()
pipe.start()