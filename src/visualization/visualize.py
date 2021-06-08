from src.settings import MODEL_PATH, FIGURE_PATH
from src.models.model import MyAwesomeModel

import matplotlib.pyplot as plt
import os
from src.data.make_dataset import mnist
from sklearn.manifold import TSNE
import sys
import argparse
import torch
import seaborn as sns
sns.set_theme()


class visualize(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for visualizing,",
            usage="python visualize.py <command>"
        )

        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def TSNE_viz(self):

        print("Visualizing numbers using t-SNE")
        parser = argparse.ArgumentParser(description='Visualization arguments')
        parser.add_argument('--name', default='base_model', type=str)
        parser.add_argument('--dataset', default='test', type=str)
        parser.add_argument('--bs', default=64, type=int)
        parser.add_argument('--show_plot', default=False, type=bool)

        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])

        if args.name:
            state_dict = torch.load(os.path.join(MODEL_PATH, args.name+'.pth'))
            model = MyAwesomeModel()
            model.load_state_dict(state_dict)

        if args.dataset == 'train':
            dataset, _ = mnist(batch_size=args.bs)

        elif args.dataset == 'test':
            _, dataset = mnist(batch_size=args.bs)
        else:
            print('Unrecognized argument. Choose either "train" or "test".')

        with torch.no_grad():
            model.eval()

            all_output = []
            all_labels = []

            for images, labels in dataset:

                output = model(images)

                all_output.append(output)
                all_labels.append(labels)

            all_output = torch.cat(tuple(all_output), 0)
            all_labels = torch.cat(tuple(all_labels), 0)

        X = all_output.numpy()

        X_embedded = TSNE(n_components=2).fit_transform(X)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        sns.scatterplot(x=X_embedded[:, 0],
                        y=X_embedded[:, 1],
                        hue=all_labels.numpy(),
                        palette=sns.hls_palette(10),
                        legend='full',
                        ax=ax)
        plt.title(f't-SNE embedding for {args.name} output for {args.dataset} set')
        plt.xlabel('Embedding dimension 1')
        plt.ylabel('Embedding dimension 2')

        fig.savefig(os.path.join(FIGURE_PATH, 't-SNE-visualizations', f't-SNE_for_{args.name}_for_{args.dataset}.png'))

        if args.show_plot:
            plt.show()
            plt.close()


if __name__ == '__main__':
    visualize()
