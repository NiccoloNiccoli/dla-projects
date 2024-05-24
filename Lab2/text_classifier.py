# Exercise 3.1 lab 2
import torch.nn as nn
import torch
from transformers import DistilBertTokenizer, DistilBertModel, set_seed
from datasets import load_dataset
from tqdm import tqdm
import wandb
import argparse

class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, text, device):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = inputs.to(device)
        output = self.model(**inputs)
        cls_token = output['last_hidden_state'][:, 0, :]
        return self.head(cls_token)

def train_head(model, train_dataset, epochs=10, lr=3e-4, device = "cpu", use_wandb=False):
    model.train()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        acc_loss = 0
        for i, batch in tqdm(enumerate(train_dataset)):
            text = batch["text"]
            label = batch["label"]
            label = label.to(device)
            optimizer.zero_grad()
            output = model(text, device)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            #accumulate the loss
            acc_loss += loss.item()
            if (i + 1) % 500 == 0:
                if use_wandb:
                    wandb.log({"loss": acc_loss / 500})
                print(f"Epoch {epoch}, batch {i}, loss {acc_loss / 500}")
                acc_loss = 0

def finetune(model, train_dataset, epochs=10, lr=3e-4, device = "cpu", use_wandb=False):
    model.train()
    # use different learning rate for the bert model
    params = [{"params": model.model.parameters(), "lr" : lr/100},
              {"params": model.head.parameters(), "lr": lr}
              ]
    optimizer = torch.optim.AdamW(params)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        acc_loss = 0
        for i, batch in tqdm(enumerate(train_dataset)):
            text = batch["text"]
            label = batch["label"]
            label = label.to(device)
            optimizer.zero_grad()
            output = model(text, device)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            #accumulate the loss
            acc_loss += loss.item()
            if (i + 1) % 500 == 0:
                if use_wandb:
                    wandb.log({"loss": acc_loss / 500})
                print(f"Epoch {epoch}, batch {i}, loss {acc_loss / 500}")
                acc_loss = 0

def test(model, dataset, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataset)):
            text = batch["text"]
            label = batch["label"]
            label = label.to(device)
            output = model(text, device)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct, total



if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wandb", action="store_true")
    args = argparser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imdb = load_dataset("imdb")
    imdb_train = imdb["train"]
    imdb_test = imdb["test"]
    dl_train = torch.utils.data.DataLoader(imdb_train, batch_size=8, shuffle = True)
    dl_test = torch.utils.data.DataLoader(imdb_test, batch_size=8, shuffle = False)
    model = TextClassifier()
    model = model.to(device)
    epochs = 1

    # zero shot accuracy
    if args.wandb:
        strategy = "zero shot"
        wandb.init(project="lab2 text classification", name = strategy, config = {"strategy": strategy})
    correct, total = test(model, dl_test, device=device)
    print(f"Zero shot accuracy: {correct/total}")
    if args.wandb:
        wandb.log({"accuracy": correct/total})
        wandb.finish()
        
    # training the head
    if args.wandb:
        strategy = "training the head"
        wandb.init(project="lab2 text classification", name = strategy, config = {"strategy": strategy})
    train_head(model, dl_train, epochs=1, device=device, use_wandb=args.wandb)
    correct, total = test(model, dl_test, device=device)
    print(f"Accuracy after training the head: {correct/total}")
    if args.wandb:
        wandb.log({"accuracy": correct/total})
        wandb.finish()

    # finetuning the model
    model = TextClassifier()
    model = model.to(device)
    if args.wandb:
        strategy = "finetuning"
        wandb.init(project="lab2 text classification", name = strategy, config = {"strategy": strategy})
    finetune(model, dl_train, epochs=1, device=device, use_wandb=args.wandb)
    correct, total = test(model, dl_test, device=device)
    print(f"Accuracy after finetuning: {correct/total}")
    if args.wandb:
        wandb.log({"accuracy": correct/total})
        wandb.finish()

    print("done")

