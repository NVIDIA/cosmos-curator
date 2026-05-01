## Pipelines

Each pipeline should
- create a list of pipeline tasks, each of which should inherit `PipelineTask` defined in [stage_interface.py](../core/interfaces/stage_interface.py).
- define a list of stages, each of which should inherit `CuratorStage` defined in [stage_interface.py](../core/interfaces/stage_interface.py).
- call `run_pipeline` defined in [pipeline_interface.py](../core/interfaces/pipeline_interface.py).