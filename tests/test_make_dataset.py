from src.data.make_dataset import mnist
import torch
import pytest


@pytest.fixture
def mnist_wrap():
    return mnist()


class TestDataset:

    def test_len(self, mnist_wrap):

        trainset, testset = mnist_wrap

        assert len(trainset) == 60000, "trainset most have length 60000"
        assert len(testset) == 10000, "testset most have length of 10000"

    def test_mnist_shape(self, mnist_wrap):

        trainset, testset = mnist_wrap

        assert trainset.tensors[0].size()[1:] == (
            1,
            28,
            28,
        ), " shape of single image must be (1,28,28)"
        assert testset.tensors[0].size()[1:] == (
            1,
            28,
            28,
        ), " shape of single image must be (1,28,28)"

    def test_all_labels_represented(self, mnist_wrap):

        trainset, testset = mnist_wrap

        train_labels = trainset.tensors[1]
        test_labels = testset.tensors[1]

        corr_labels = torch.arange(0, 10)

        assert torch.all(
            torch.unique(train_labels) == corr_labels
        ), "Not all labels are present in train set"
        assert torch.all(
            torch.unique(test_labels) == corr_labels
        ), "Not all labels are present in test set"
