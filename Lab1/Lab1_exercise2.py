from models import FullyConvolutionalNN
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import wandb
from modifiedMNIST import ModifiedMNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Lab 1 of Deep Learning Applications: Exercise 2.2")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--translate", type = float, default = 0.4, help = "Translate factor for data augmentation")
    parser.add_argument("--load-model", type=str, help="Load model from file", default=None)
    return parser.parse_args()

def train_one_epoch(model, dataloader, optimizer, epoch, dataset_name = "Unknown", device = 'cpu', use_wandb = False):
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc=f'Training epoch {epoch}'):
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        _, _, a, b = logits.shape
        gts = y.unsqueeze(-1).unsqueeze(-1).repeat(1,a,b)
        loss = F.cross_entropy(logits, gts)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if use_wandb:
        wandb.log({f'train_loss_{dataset_name}': np.mean(losses)}, step=epoch)
    return np.mean(losses), np.std(losses)

def evaluate_model(model, dataloader, epoch, dataset_name = "Unknown", is_test = False, device = 'cpu', use_wandb = False):
    model.eval()
    gts = []
    accuracy = 0.0
    total_elements = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating @ {epoch}'):
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, preds = torch.max(logits, 1)
            batch_size, a, b = preds.shape
            gts = y.unsqueeze(-1).unsqueeze(-1).repeat(1,a,b).to(device)
            correct_elements = torch.eq(preds, gts)
            accuracy += correct_elements.sum().item()
            total_elements += batch_size * a * b
        accuracy /= total_elements
    if use_wandb:
        if is_test:
            wandb.log({f'test_accuracy_{dataset_name}': accuracy})
        else:
            wandb.log({f'val_accuracy_{dataset_name}': accuracy})
    return accuracy        

if __name__ == "__main__":
    '''
    Using a Fully Convolutional Neural Network to classify MNIST images.
    '''
    args = parse_args()

    # set seed
    torch.manual_seed(7)

    # Load MNIST
    transform = transforms.Compose([
        transforms.RandomAffine(0, translate=(args.translate, args.translate)), # this transformation is VERY important 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
     # create validation set 80 : 20
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define model and optimizer
    model = FullyConvolutionalNN(input_channels = 1, output_channels = 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # initialize wandb
    if args.wandb:
        config = {
            'model' : "FullyConvolutionalNN",
            'dataset' : "MNIST",
            'epochs' : args.epochs,
            'batch_size' : args.batch_size,
            'lr' : args.lr,
            'optimizer' : 'AdamW'
        }
        wandb.init(project="DLA-Lab1-Ex2.2", name = f"FullyConvolutionalNN_MNIST", config = config)
    
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    else:
        # train the model
        train_losses_mean, train_losses_std = [], []
        val_accuracies = []
        for epoch in range(args.epochs):
            train_loss, train_loss_std = train_one_epoch(model, train_loader, optimizer, epoch = epoch, dataset_name="MNIST", device = device, use_wandb = args.wandb)
            train_losses_mean.append(train_loss)
            train_losses_std.append(train_loss_std)
            val_acc = evaluate_model(model, val_loader, epoch=epoch, dataset_name="MNIST", is_test=False, device = device, use_wandb = args.wandb)
            val_accuracies.append(val_acc)
            print(f'Epoch {epoch} : Train Loss = {train_loss} +/- {train_loss_std}, Val Accuracy = {val_acc}')
        
        # test on MNIST
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        test_acc = evaluate_model(model, test_loader, epoch=0, dataset_name="MNIST", is_test=True, device = device, use_wandb = args.wandb)
        print(f'Test Accuracy on MNIST: {test_acc}')

        # save model
        torch.save(model.state_dict(), 'fully_convolutional_nn_mnist.pth')

    # load test data (Modified MNIST)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    m_test_dataset = ModifiedMNIST(root='./data', train=True, download=True, transform=test_transform)
    m_test_loader = DataLoader(m_test_dataset, batch_size=512, shuffle=True)


    # test on Modified MNIST
    accuracy = 0.0
    model.eval()
    total_elements = 2
    for x,y1, y2 in m_test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        for i, p in enumerate(preds):
            unique_elements, counts = torch.unique(p, return_counts=True)
            sorted_counts = torch.argsort(counts, descending=True)
            accuracy += torch.sum(torch.tensor([y1[i], y2[i]]) == unique_elements[sorted_counts[:2]].cpu())/2
    print(f'Test Accuracy on Modified MNIST: {accuracy/len(m_test_dataset)}')

    
    # print image from test set
    for i in range(1):
        img, _, _ = m_test_dataset[i]
        img = img.unsqueeze(0).to(device)
        output = model(img)
        pred = torch.argmax(output, 1).squeeze()
        plt.figure(figsize=(5,5))
        plt.imshow(img.cpu().squeeze().numpy(), cmap = 'gray')

        plt.savefig(f'example.png')

        plt.figure(figsize=(5,5))
        padding = img.shape[2]//pred.shape[1]
        plt.imshow(img.cpu().squeeze().numpy(), cmap = 'gray')
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                plt.text(j * padding, i * padding, ".",ha='center', va='center', color = '#00ffff', fontsize = 12)
                plt.text(j * padding + 4, i * padding + 4, str(pred[i,j].item()),ha='center', va='center', color = 'red', fontsize = 12)
        plt.savefig(f'prediction.png')
        break
    
    if args.wandb:
        wandb.finish()

