# Exercise 1
from torch.utils.data import Dataset
import tiktoken
import argparse
from dataclasses import dataclass
from GPT import GPT
import torch
from tqdm import tqdm
import random
import wandb

class DanteSet(Dataset):
    def __init__(self, src_path, train=True):
        with open(src_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
        self.enc = tiktoken.get_encoding("gpt2")
        # self.t_p = 0.9
        self.enc = tiktoken.get_encoding("gpt2")
        self.ids = self.enc.encode_ordinary(self.data)
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.ids[(idx+1)%len(self.ids)]

def make_batch(dataset, batch_size, seq_length, shuffle=True, targets=True):
    batch = torch.zeros(batch_size, seq_length, dtype=torch.long)
    if targets:
        target_batch = torch.zeros(batch_size, seq_length, dtype=torch.long)
    else:
        target_batch = None
    for b in range(batch_size):
        starting_index = b
        if shuffle:
            starting_index = random.randint(0, len(dataset)-seq_length)
        for seq in range(seq_length):
            sequence = dataset[starting_index+seq]
            batch[b][seq] = sequence[0]
            if targets:
                target_batch[b][seq] = sequence[1]
    return batch, target_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=str, default="inferno.txt")
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(7)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = DanteSet(args.src_path, train=True)

    # instatiating the model
    @dataclass
    class Config:
        n_embd: int = 768
        block_size: int = 16
        vocab_size: int = 50304
        bias: bool = False
        dropout: float = 0.0
        n_layer: int = 12
        n_head: int = 12
    
    config = Config()

    model = GPT(config).to(device)

    # defining the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

    # training the model
    if args.use_wandb:
        wandb_config = {
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
            "bias": config.bias,
            "dropout": config.dropout,
            "n_layer": config.n_layer,
            "n_head": config.n_head
        }
        wandb.init(project="Lab2 Dante-like Text Generator", config=wandb_config)
    iterations = int(1e5)
    shuffle = True
    model.train()
    for i in tqdm(range(iterations)):
        x, y = make_batch(ds, 4, config.block_size, shuffle)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if args.use_wandb:
            wandb.log({"loss": loss.item()})
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"Iteration {i+1} : {ds.enc.decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=config.block_size).tolist()[0])}")
    
    # save the model
    torch.save(model.state_dict(), "dante_model.pth")

    with open("output_dante.txt", "w") as f:
        f.write(ds.enc.decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=1024).tolist()[0]))

    if args.use_wandb:
        wandb.finish()

