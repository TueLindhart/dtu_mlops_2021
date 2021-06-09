from src.models.model import MyAwesomeModel
import torch
import torch.nn.functional as F
import pytest


def model():
    return MyAwesomeModel()


class TestModel:
    def test_output_shape(self):
        x = torch.randn(1, 1, 28, 28)

        model = MyAwesomeModel()
        output = model(x)

        assert output.size() == (1, 10), "Output must have shape (1,10)"

    def test_sum_to_1(self):

        model = MyAwesomeModel()
        x = torch.randn(1, 1, 28, 28)
        output = model(x)

        sum_ = torch.sum(F.softmax(output, dim=1)).data.numpy()

        assert (0.999 <= sum_) and (sum_ <= 1.001), "Softmax output must sum to one"

    def test_num_dim_raise(self):

        model = MyAwesomeModel()

        x = torch.randn(1, 1, 1, 28, 28)

        with pytest.raises(ValueError):
            model(x)

    def test_wrong_dim_raise(self):

        model = MyAwesomeModel()

        x = torch.randn(1, 1, 30, 28)

        with pytest.raises(ValueError):
            model(x)

    @pytest.mark.parametrize(
        "x, error_type",
        [
            (torch.randn(1, 1, 1, 28, 28), ValueError), 
            (torch.randn(1, 28, 28), ValueError), 
            (torch.randn(1, 3, 28, 28), ValueError), 
            (torch.randn(1, 1, 30, 28), ValueError), 
            (torch.randn(1, 1, 30, 28), ValueError), 
        ],
    )
    def test_dim_raise(self, x, error_type):

        model = MyAwesomeModel()

        with pytest.raises(error_type):
            model(x)
