from models import NonSizeDependentCNN
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from modifiedMNIST import ModifiedMNIST
import numpy as np
from Lab1_exercise1 import train_one_epoch, evaluate_model
from PIL import Image
from torch.utils.data import Dataset

class MultiSizeMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, image_size=32):
        super().__init__()
        self.ds = MNIST(root, train, transform, target_transform, download)
        self.transform = transform
        self.image_size = image_size

    def __getitem__(self, index):
        img, target = self.ds[index]
        #make a new image and place the original image somewhere in the new image
        new_img = Image.new('1', (self.image_size, self.image_size), 0)
        convert_to_pil = transforms.ToPILImage()
        new_img = self.transform(new_img)
        #sum the two tensors but img is shifted by 16 pixels
        #pick two random numbers between 0 and 32
        x = np.random.randint(0, self.image_size - 28)
        y = np.random.randint(0, self.image_size - 28)
        #sum two tensors of different shape
       # print(new_img.shape, img.shape)
        new_img[:,x:x+28, y:y+28] = img
        return new_img, target
    def __len__(self):
        return len(self.ds)


def parse_args():
    parser = argparse.ArgumentParser(description="Lab 1 of Deep Learning Applications: Training and testing a CNN that does not depend on the size of the input")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--translate", type = float, default = 0.4, help = "Translate factor for data augmentation")
    parser.add_argument("--load-model", type=str, help="Load model from file", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set the seed
    torch.manual_seed(7)

    # set the transforms
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(args.translate, args.translate)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load the dataset
    train_dataset = MNIST('data', train=True, download=True, transform=transform)

    # create validation set 80 : 20
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # create the model
    model = NonSizeDependentCNN(1, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # initialize wandb
    if args.wandb:
        config = {
            'model' : 'NonSizeDependentCNN',
            'dataset' : 'MNIST',
            'epochs' : args.epochs,
            'batch_size' : args.batch_size,
            'lr' : args.lr,
            'optimizer' : 'AdamW'
        }
        wandb.init(project="DLA-Lab1-Exercise2.2-Non-Size-Dep-CNN", name = "NonSizeDependentCNN", config=config)

    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    else:
        train_losses_mean, train_losses_std = [], []
        val_accuracies = []
        for epoch in range(args.epochs):
            train_loss_mean, train_loss_std = train_one_epoch(model, train_loader, optimizer, epoch = epoch, dataset_name="MNIST", device = device, use_wandb = args.wandb)
            val_accuracy = evaluate_model(model, val_loader, epoch=epoch, dataset_name="MNIST", is_test=False, device = device, use_wandb = args.wandb)
            train_losses_mean.append(train_loss_mean)
            train_losses_std.append(train_loss_std)
            val_accuracies.append(val_accuracy)
            print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss_mean} +/- {train_loss_std} - Val Accuracy: {val_accuracy}')

    # test on MultiSizeMNIST
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = MultiSizeMNIST('data', train=False, download=True, transform=test_transform, image_size=64)
    img, target = test_dataset[0]
    img = img.permute(1,2,0)
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.title(target)
    plt.savefig('multisizeMNIST.png')

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    accuracy = evaluate_model(model, test_dataloader, epoch=0, dataset_name="MultiSizeMNIST", is_test=True, device = device, use_wandb = args.wandb)
    print(f'Test Accuracy on MultiSizeMNIST: {accuracy}')

    if args.wandb:
        wandb.finish()


