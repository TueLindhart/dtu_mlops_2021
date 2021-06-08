import sys
import os
import argparse

from src.models.model import MyAwesomeModel
from src.settings import MODEL_PATH

import torch
import torch.nn.functional as F
from src.data.make_dataset import mnist


def predict():
    parser = argparse.ArgumentParser(description='Prediction arguments')
    parser.add_argument('--name', default='base_model', type=str)
    parser.add_argument('--bs', default=64, type=int)

    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    if args.name:
        state_dict = torch.load(os.path.join(MODEL_PATH, args.name+'.pth'))
        model = MyAwesomeModel()
        model.load_state_dict(state_dict)

    _, test_set = mnist(batch_size=args.bs)

    with torch.no_grad():

        model.eval()

        all_predictions = []

        for images, labels in test_set:

            ps = F.softmax((model(images)), dim=1)

            top_p, top_class = ps.topk(1, dim=1)

            all_predictions.append(top_class)

        all_predictions = torch.cat(tuple(all_predictions), 0)

    return all_predictions
