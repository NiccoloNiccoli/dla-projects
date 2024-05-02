import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import models as m
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    return parser.parse_args()

# train one epoch
def train_one_epoch(model, dataloader, optimizer, epoch , dataset_name = "Unknown", device = 'cpu', use_wandb = False):
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc=f'Training epoch {epoch}'):
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # log the loss
    if use_wandb:
        wandb.log({f'train_loss_{dataset_name}': np.mean(losses)}, step=epoch)
    return np.mean(losses), np.std(losses)


# evaluate the model
def evaluate_model(model, dataloader, epoch, dataset_name = "Unknown", is_test = False,  device = 'cpu', use_wandb = False):
    model.eval()
    with torch.no_grad():
        correct = 0
        for batch in tqdm(dataloader, desc=f'Evaluating @ {epoch}'):
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == y).item()
    accuracy = correct / len(dataloader.dataset)
    if use_wandb:
        if is_test:
            wandb.log({f'test_accuracy_{dataset_name}': accuracy})
        else:
            wandb.log({f'val_accuracy_{dataset_name}': accuracy}, step = epoch)
    return accuracy

if __name__=="__main__":
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # compare various models on mnist and cifar10
    # 1. Basic MLP - 2 layers
    # 2. More complex MLP - 4 layers
    # 3. CNN - 4 conv layers + 2 linear layers

    datasets = {
        "MNIST": MNIST,
        "CIFAR10": CIFAR10
    }
    for (dataset_name, dataset) in datasets.items():
        # set seed
        torch.manual_seed(7)

        # load dataset
        if dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            models = {
                "Basic MLP": m.MLP(28*28, 256, 2, 10).to(device),
                "Complex MLP": m.MLP(28*28, 256, 4, 10).to(device),	
                "CNN": m.CNN(1, hidden_dim_mlp=2048, output_dim=10).to(device),
            }
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            models = {
                "Basic MLP": m.MLP(32*32*3, 256, 2, 10).to(device),
                "Complex MLP": m.MLP(32*32*3, 256, 4, 10).to(device),	
                "CNN": m.CNN(3, hidden_dim_mlp=4608, output_dim=10).to(device),
            }
        train_dataset = dataset(root='./data', train=True, download=True, transform=transform)

        # create validation set 80 : 20
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        test_dataset = dataset(root='./data', train=False, download=True, transform=transform)

        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        for (model_name, model) in models.items():
            
            print(f'Training {model_name} on {dataset_name}')

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

            # initialize wandb
            if args.wandb:
                config = {
                    'model' : model_name,
                    'dataset' : dataset_name,
                    'epochs' : args.epochs,
                    'batch_size' : args.batch_size,
                    'lr' : args.lr,
                    'optimizer' : 'AdamW'
                }

                wandb.init(project='DLA-Lab1', config=config, name=f'{model_name}_{dataset_name}')

            # train the model
            train_losses_mean, train_losses_std = [], []
            val_accuracies = []
            for epoch in range(args.epochs):
                train_loss, train_loss_std = train_one_epoch(model, train_loader, optimizer, epoch, dataset_name=dataset_name, device=device, use_wandb=args.wandb)
                train_losses_mean.append(train_loss)
                train_losses_std.append(train_loss_std)
                val_accuracy = evaluate_model(model, val_loader, epoch, dataset_name=dataset_name, device=device, use_wandb=args.wandb, is_test=False)
                val_accuracies.append(val_accuracy)
                print(f'Epoch {epoch} : Train Loss = {train_loss} +/- {train_loss_std}, Val Accuracy = {val_accuracy}')
            
            # evaluate on test set
            test_accuracy = evaluate_model(model, test_loader, epoch, dataset_name=dataset_name, device=device, use_wandb=args.wandb, is_test=True)
            print(f'Test Accuracy = {test_accuracy}')

            # plot the training loss and validation accuracy
            plt.figure(figsize=(10,5))
            x = np.arange(config['epochs'])
            plt.plot(x, train_losses_mean, label='Train loss')
            plt.fill_between(x, np.array(train_losses_mean) - np.array(train_losses_std), np.array(train_losses_mean) + np.array(train_losses_std), color='#0066FF', alpha=0.3, label='±1 std')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Train loss of {model_name} on {dataset_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'./train_loss_{model_name}_{dataset_name}.png')

            plt.figure(figsize=(10,5))
            plt.plot(val_accuracies, label='Val accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Validation accuracy of {model_name} on {dataset_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'./val_accuracy_{model_name}_{dataset_name}.png')

            if args.wandb:
                wandb.finish()
            


        