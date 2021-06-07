import sys
import os
import argparse

import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from data import mnist
from model import MyAwesomeModel


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
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=1e-4,type=float)
        parser.add_argument('--e',default=10,type=int)
        parser.add_argument('--name',default='base_model',type=str)


        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)


        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()

        # Define criterion
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        lr = args.lr
        optimizer = optim.Adam(model.parameters(),lr=args.lr)

        epochs = args.e

        running_loss_epoch = []

        for e in range(epochs):

            running_loss = 0
            model.train()
            
            for images, labels in train_set:

                # Make gradients zero
                optimizer.zero_grad()

                # Run through model
                out = model(images)
                #Calculate loss
                loss = criterion(out,labels)
                #Calculate gradients
                loss.backward()

                # Optimize use SGD
                optimizer.step()

                running_loss += loss.item()
                
            else:
                running_loss = running_loss/len(train_set)
                running_loss_epoch.append(running_loss)

                print(f"Epoch {e+1}, Training loss: {running_loss}")
        else:

            # Save model
            torch.save(model.state_dict(),f'models/{args.name}.pth')

            # Plot training progress
            plt.figure(figsize=(5,5))
            plt.plot(range(1,e+2),running_loss_epoch)
            plt.xlabel('Epochs')
            plt.ylabel('Cross-entropy loss')
            plt.title('Training loss for model')
            #plt.savefig(f'~/Desktop/MLOps/dtu_mlops/01_introduction/final_exercise/training_figures/{args.name}.png')
            plt.savefig(f'training_figures/{args.name}.png')
            plt.show()
            plt.close()

        
    def eval(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="models/",type=str)
        parser.add_argument('--name',default='base_model',type=str)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.name:
            state_dict = torch.load(args.load_model_from+args.name+'.pth')
            model = MyAwesomeModel()
            model.load_state_dict(state_dict)

        _, test_set = mnist()

        with torch.no_grad():

            model.eval()

            all_top_classes = []
            all_labels = []

            for images, labels in test_set:

                ps = torch.exp(model(images))

                top_p, top_class = ps.topk(1, dim=1)

                all_top_classes.append(top_class)
                all_labels.append(labels)

                
            else:
                # Make all saved predictions and labels into one long tensor
                all_top_classes = torch.cat(tuple(all_top_classes),0)
                all_labels = torch.cat(tuple(all_labels))

                # Calculate accuracy
                equals = all_top_classes == all_labels.view(*all_top_classes.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                
                # Print validation accuracy of model
                print(f'Validation accuracy: {accuracy.item()*100}%')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    