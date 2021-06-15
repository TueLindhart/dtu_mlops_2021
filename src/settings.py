import os

MODULE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir))
MODULE_PATH = './'

TRAIN_FIGURE_PATH = os.path.join(
    MODULE_PATH, 'reports', 'figures', 'training_plots')
MODEL_PATH = os.path.join(MODULE_PATH, 'models')
AZURE_OUTPUT_PATH = os.path.join(MODULE_PATH, 'azure_output')

MNIST_SERVE_SCRIPT = os.path.join(MODULE_PATH, 'src', 'models', 'mnist_serve', 'mnist_script_config.py')
