import sys
import argparse

from src.models import predict_model, train_model


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )

        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if hasattr(train_model, args.command):
            # use dispatch pattern to invoke method with same name
            getattr(train_model, args.command)()

        elif hasattr(predict_model, args.command):
            getattr(predict_model, args.command)()

        else:
            print('Unrecognized command')
            parser.print_help()
            exit(1)


if __name__ == '__main__':
    TrainOREvaluate()
