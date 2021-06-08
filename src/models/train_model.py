import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel
from src.settings import FIGURE_PATH, MODEL_PATH

sns.set_theme()


def train():
    print("Training day and night")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--e", default=10, type=int)
    parser.add_argument("--name", default="base_model", type=str)
    parser.add_argument("--show_plot", default=False, type=bool)
    parser.add_argument("--tb_name", default='def', type=str)

    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, test_set = mnist()

    writer = SummaryWriter(log_dir=os.path.join("runs", args.tb_name))
    # writer = SummaryWriter()
    images, labels = next(iter(train_set))
    grid = torchvision.utils.make_grid(images)
    writer.add_image("images", grid, 0)
    writer.add_graph(model, images)
    writer.close()

    # Define criterion
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.e

    running_loss_epoch = []
    running_eval_loss_epoch = []

    for e in range(epochs):

        with torch.no_grad():

            model.eval()

            running_loss = 0

            for images, labels in test_set:

                # Run through model
                out = model(images)
                # Calculate loss
                loss = criterion(out, labels)

                # add validation loss
                running_loss += loss.item()

            else:
                running_loss = running_loss / len(test_set)
                running_eval_loss_epoch.append(running_loss)
                writer.add_scalar("Loss/test", running_loss, e + 1)  # change to add_scalars
                writer.close()
                print(f"Epoch {e+1}, Validation loss: {running_loss}")

        running_loss = 0
        model.train()

        for images, labels in train_set:

            # Make gradients zero
            optimizer.zero_grad()

            # Run through model
            out = model(images)
            # Calculate loss
            loss = criterion(out, labels)
            # Calculate gradients
            loss.backward()

            # Optimize use SGD
            optimizer.step()

            running_loss += loss.item()

        else:
            running_loss = running_loss / len(train_set)
            running_loss_epoch.append(running_loss)

            writer.add_scalar("Loss/train", running_loss, e + 1)  # change to add_scalars
            writer.close()
            print(f"Epoch {e+1}, Training loss: {running_loss}")
            print("")

            # Add histogram for weights
            writer.add_histogram("Weights/conv1.bias", model.conv1.bias, e + 1)
            writer.add_histogram("Weights/conv1.weight", model.conv1.weight, e + 1)
            writer.add_histogram("Weights/conv2.bias", model.conv2.bias, e + 1)
            writer.add_histogram("Weights/conv2.weight", model.conv2.weight, e + 1)
            writer.add_histogram("Weights/fc1.weight", model.fc1.weight, e + 1)
            writer.add_histogram("Weights/fc1.bias", model.fc1.bias, e + 1)
            writer.add_histogram("Weights/fc2.weight", model.fc2.weight, e + 1)
            writer.add_histogram("Weights/fc2.bias", model.fc2.bias, e + 1)
            writer.add_histogram("Weights/fc3.weight", model.fc3.weight, e + 1)
            writer.add_histogram("Weights/fc3.bias", model.fc3.bias, e + 1)
            writer.close()

    else:

        # Save model
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{args.name}.pth"))

        last_train_loss = running_loss_epoch[-1]
        last_eval_loss = running_eval_loss_epoch[-1]

        writer.add_hparams({'lr': args.lr, 'epochs': args.e},
                           {'hparam/train_loss': last_train_loss,
                            'hparam/eval_loss': last_eval_loss})
        writer.close()

        # Plot training progress
        plt.figure(figsize=(5, 5))
        plt.plot(range(1, e + 2), running_loss_epoch, label="Train loss")
        plt.plot(range(1, e + 2), running_eval_loss_epoch, label="Validation loss")
        plt.ylim(0, 0.5)
        plt.xlabel("Epochs")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"Training loss for model {args.name}")
        plt.legend()
        plt.savefig(os.path.join(FIGURE_PATH, "training_plots", f"{args.name}.png"))

        if args.show_plot:
            plt.show()
            plt.close()


def eval():
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("--load_model_from", default=MODEL_PATH, type=str)
    parser.add_argument("--name", default="base_model", type=str)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    # TODO: Implement evaluation logic here
    if args.name:
        state_dict = torch.load(os.path.join(args.load_model_from, args.name + ".pth"))
        model = MyAwesomeModel()
        model.load_state_dict(state_dict)

    _, test_set = mnist()

    with torch.no_grad():

        model.eval()

        all_top_classes = []
        all_labels = []

        for images, labels in test_set:

            # ps = torch.exp(model(images))
            ps = F.softmax(model(images), dim=1)

            top_p, top_class = ps.topk(1, dim=1)

            all_top_classes.append(top_class)
            all_labels.append(labels)

        else:
            # Make all saved predictions and labels into one long tensor
            all_top_classes = torch.cat(tuple(all_top_classes), 0)
            all_labels = torch.cat(tuple(all_labels))

            # Calculate accuracy
            equals = all_top_classes == all_labels.view(*all_top_classes.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))

            # Print validation accuracy of model
            print(f"Validation accuracy: {accuracy.item()*100}%")
