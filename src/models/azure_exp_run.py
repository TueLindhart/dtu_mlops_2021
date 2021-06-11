from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.widgets import RunDetails

from src.settings import MODULE_PATH, MODEL_PATH
import os

from azureml.core import Workspace

# from azureml.core.compute import ComputeTarget

ws = Workspace.from_config(os.path.join(MODULE_PATH, 'config.json'))
print("------",MODULE_PATH)

# From a pip requirements file
myenv = Environment.from_pip_requirements(name="azure_venv",
                                          file_path=os.path.join(MODULE_PATH, "requirements2.txt"))

myenv.register(workspace=ws)

# myenv = Environment.get(workspace=ws, name="azure_venv")
# myenv = Environment("user-managed-env")
# myenv.python.user_managed_dependencies = './venv/bin/python'

# my_compute_target = ComputeTarget(workspace=ws, name='compuInstTwo')
my_compute_target = ws.compute_targets['compuInstTwo']
# my_compute_target = "compuInstTwo"

# Create a script config
script_config = ScriptRunConfig(source_directory=os.path.join(MODULE_PATH, 'src', 'models'),
                                script='main.py',
                                compute_target=my_compute_target,
                                arguments=["train", "--lr", 1e-4, "--e",
                                           10, "--bs", 128, "--name", "azure_exp_1"],
                                environment=myenv)

script_config.run_config.target = my_compute_target

# submit the experiment
experiment_name = 'mnist_exp_1'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)
#RunDetails(run).show()
run.wait_for_completion(show_output=True)
