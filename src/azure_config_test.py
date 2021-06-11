from src.settings import MODULE_PATH

from azureml.core import Workspace
import os

ws = Workspace.from_config(os.path.join(MODULE_PATH, 'config.json'))
