from src.data.make_dataset import mnist
from azureml.core import Environment
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core.model import InferenceConfig, Model
# from azureml.core.conda_dependencies import CondaDependencies
from src.settings import MODULE_PATH, MNIST_SERVE_SCRIPT
import os
import json

ws = Workspace.from_config(os.path.join(MODULE_PATH, 'config.json'))
model = ws.models['mnist_model']


myenv = Environment.get(workspace=ws, name="azure_venv")
# conda_dep = CondaDependencies()
# conda_dep.add_pip_package("azureml-defaults")
# myenv.python.conda_dependencies = conda_dep
# myenv.register(workspace=ws)

# Configure the scoring environment
inference_config = InferenceConfig(source_directory=MODULE_PATH,
                                   entry_script=MNIST_SERVE_SCRIPT,
                                   environment=myenv)

# cluster_name = 'aks-cluster'
# compute_config = AksCompute.provisioning_configuration(location='northeurope')
# production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)


deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1)

service_name = "mnist-service"

production_cluster = ws.compute_targets['GPUmlops']

service = Model.deploy(
    workspace=ws, name=service_name, models=[
        model], inference_config=inference_config,
    deployment_config=deployment_config, deployment_target=production_cluster)

service.wait_for_deployment(True)
print(service.state)
print(service.get_logs())

# Test deployment

_, test_loader = mnist(tensor_in_ram=False)

images, labels = next(iter(test_loader))

image_batch = images.numpy()

input_json = json.dumps({"data": image_batch})

predictions = service.run(input_data=input_json)

predicted_classes = json.loads(predictions)

print(predicted_classes)
