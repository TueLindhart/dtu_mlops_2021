from src.data.make_dataset import mnist

_, test_loader = mnist(tensor_in_ram=False)

images, labels = next(iter(test_loader))

image_batch = images.numpy()
