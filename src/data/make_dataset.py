# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms

from src.settings import MODULE_PATH


def mnist(batch_size=64, tensor_in_ram=True):
    # exchange with the real mnist dataset

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Download and load the training data
    trainset = datasets.MNIST(os.path.join(MODULE_PATH, 'data'), download=True, train=True,
                              transform=transform)
    testset = datasets.MNIST(os.path.join(MODULE_PATH, 'data'), download=True, train=False,
                             transform=transform)

    if tensor_in_ram:
        train_d = trainset.data
        train_t = trainset.targets

        test_d = testset.data
        test_t = testset.targets

        trainset = torch.utils.data.Dataset(train_d, train_t)
        testset = torch.utils.data.Dataset(test_d, test_t)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    mnist()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
