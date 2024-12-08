import torch
from gpt import GPTLanguageModel
import wandb
import os

# Training Hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------


# Initialize wandb
wandb.init(
    project="gpt",
    config={
        "batch_size": batch_size,
        "block_size": block_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "learning_rate": learning_rate,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }
)

torch.manual_seed(1337)

# Fetch Input
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# ------------

# Tokenization
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string
# -------------

# Data Loading
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# -------------

# Training


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        # Log metrics to wandb
        wandb.log({
            "train_loss": losses['train'],
            "val_loss": losses['val'],
            "iteration": iter
        })

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Create a directory for checkpoints
os.makedirs('checkpoints', exist_ok=True)

# Save the model weights locally first
checkpoint_path = os.path.join('checkpoints', 'final_model.pth')
torch.save(model.state_dict(), checkpoint_path)

# Save to wandb with base_path
wandb.save(checkpoint_path, base_path=os.getcwd())

# Also save vocabulary info for later use
vocab_path = os.path.join('checkpoints', 'vocab_info.pth')
torch.save({'stoi': stoi, 'itos': itos}, vocab_path)
wandb.save(vocab_path, base_path=os.getcwd())

# Generate sample text and log it
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
wandb.log({"generated_text": generated_text})
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# Close wandb run
wandb.finish()
