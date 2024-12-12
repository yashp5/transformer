# %%
import torch
import torch.nn as nn
import math
import wandb
wandb.init(project="gpt-language-model", config={
    "batch_size": 64,
    "block_size": 256,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 6,
    "dropout": 0.2,
    "learning_rate": 5e-5,
    "max_iters": 5000,
    "eval_interval": 500,
    "eval_iters": 200,  # Add this missing parameter
    "device": "cuda" if torch.cuda.is_available() else "cpu"
})
config = wandb.config

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

# Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

# Transformer Block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = SinusoidalPositionalEmbedding(config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding(tok_emb)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            probs, indices = probs.topk(top_k, dim=-1)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            idx_next = indices.gather(-1, torch.multinomial(probs, 1))
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

vocab_size = 65




# %%
# Initialize the model architecture
model = GPTLanguageModel(vocab_size).to("cpu")

# Load the state_dict
state_dict = torch.load("final_model.pth", map_location="cpu")

# Load the weights into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully and ready for inference!")


# %%
import torch
import time
import os
import psutil
import torch.quantization as quantization
import torch.nn.functional as F


# Helper function to calculate memory usage
def get_memory_usage():
    """Return memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

# Helper function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Helper function to measure inference time
def measure_inference_time(model, input_tensor, iterations=100):
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            model(input_tensor)
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    return avg_time




# Load QAT Model
# Load QAT Model

# Dummy input for testing
input_tensor = torch.randint(0, 65, (1, 256))  # Batch size=1, Sequence length=256

# Evaluate Base Model
print("Evaluating Base Model...")
base_params = count_parameters(model)
base_memory = get_memory_usage()
base_inference_time = measure_inference_time(model, input_tensor)

# Evaluate QAT Model
print("Evaluating QAT Model...")
qat_params = count_parameters(loaded_model)
qat_memory = get_memory_usage()
qat_inference_time = measure_inference_time(loaded_model, input_tensor)

# Print Results
print("Performance Comparison")
print("======================")
print(f"Base Model Parameters: {base_params}")
print(f"QAT Model Parameters: {qat_params}")
print(f"Base Model Memory Usage: {base_memory:.2f} MB")
print(f"QAT Model Memory Usage: {qat_memory:.2f} MB")
print(f"Base Model Inference Time: {base_inference_time * 1000:.2f} ms/token")
print(f"QAT Model Inference Time: {qat_inference_time * 1000:.2f} ms/token")

# %%
for name, param in qat_model.named_parameters():
    print(name, param.dtype)


# %%
# Switch to evaluation mode

from torch.quantization import convert
# Convert the model to a quantized format
convert(qat_model, inplace=True)
print("Model converted to quantized format.")


# %%
torch.save(quantized_model, 'qat_model.pth')

# %%
# When saving the model:
torch.save(qat_model.state_dict(), "qat_model.pth")

# When loading the model:
# Make sure you have the same model architecture defined first
state_dict = torch.load("qat_model.pth", map_location="cpu")
qat_model.load_state_dict(state_dict, strict=False)
qat_model.eval()

# %%
# Method 1: Check quantization attributes
def is_model_quantized(model):
    # Check if the model has quantization-related attributes
    for name, module in model.named_modules():
        if hasattr(module, 'qconfig'):
            return True
    return False

# Method 2: Inspect module types
def check_quantization_details(model):
    quantized_modules = []
    for name, module in model.named_modules():
        # Check for known quantized module types
        quantized_types = [
            'QuantizedLinear',
            'QuantizedConv2d',
            'QLinear',
            'QConv2d'
        ]

        module_type = type(module).__name__
        if any(q_type in module_type for q_type in quantized_types):
            quantized_modules.append((name, module_type))

    return quantized_modules

# Usage
print("Is model quantized?", is_model_quantized(quantized_model))
quantized_details = check_quantization_details(quantized_model)
if quantized_details:
    print("Quantized modules:")
    for name, module_type in quantized_details:
        print(f"- {name}: {module_type}")
else:
    print("No quantized modules found.")

# Additional check for quantization configuration
if hasattr(qat_model, 'qconfig'):
    print("Model has quantization configuration")
    print("Quantization config:", qat_model.qconfig)

# %%
quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize only linear layers
        dtype=torch.qint8  # 8-bit quantization
    )

# %%
quant_params = sum(p.numel() for p in quantized_model.parameters())
original_params = sum(p.numel() for p in model.parameters())
print(original_params)

# %%
# Saving the quantized model
torch.save({
    'model_state_dict': quantized_model.state_dict(),
    'model_architecture': type(quantized_model).__name__,
    'vocab_size': vocab_size
}, 'quantized_model.pth')

# Loading the quantized model
checkpoint = torch.load('quantized_model.pth')

# Recreate the model with the same architecture
loaded_model = GPTLanguageModel(checkpoint['vocab_size']).to("cpu")

# Load the state dict with a less strict approach
loaded_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# %%
# Saving entire quantized model
torch.save(quantized_model, 'full_quantized_model.pt')

# Loading entire quantized model
loaded_model = torch.load('full_quantized_model.pt')

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset from text file
class TextDataset(Dataset):
    def __init__(self, file_path, vocab_size=10000, seq_length=50):
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Create vocabulary
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        # Encode text
        self.encoded = [self.char_to_idx[ch] for ch in text]

        # Prepare sequences
        self.sequences = []
        self.targets = []
        for i in range(0, len(self.encoded) - seq_length, seq_length):
            seq = self.encoded[i:i+seq_length]
            target = self.encoded[i+1:i+seq_length+1]

            # Pad or truncate if needed
            seq = seq[:seq_length] + [0] * max(0, seq_length - len(seq))
            target = target[:seq_length] + [0] * max(0, seq_length - len(target))

            self.sequences.append(seq)
            self.targets.append(target)

        # Convert to tensors
        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Create test loader
test_dataset = TextDataset('more.txt')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Modify the metrics computation function
def compute_model_metrics(original_model, quantized_model, test_loader):
    # Model Size Comparison
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    # Inference Time Measurement
    def measure_inference_time(model, test_loader, num_iterations=100):
        model.eval()
        total_time = 0
        with torch.no_grad():
            for _ in range(num_iterations):
                for batch, targets in test_loader:
                    start_time = time.time()
                    _ = model(batch)
                    total_time += time.time() - start_time
                    break  # Just use first batch

        avg_inference_time = total_time / num_iterations
        return avg_inference_time

    # Loss Computation (instead of accuracy for text generation)
    # def compute_loss(model, test_loader):
    #     model.eval()
    #     total_loss = 0
    #     total_batches = 0
    #     criterion = nn.CrossEntropyLoss()
    #     with torch.no_grad():
    #         for batch, targets in test_loader:
    #             outputs = model(batch)
    #             logits = outputs.logits  # Access the prediction tensor
    #             loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    #             # outputs = model(batch)
    #             # # Reshape outputs and targets for loss computation
    #             # loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    #             total_loss += loss.item()
    #             total_batches += 1

    #     avg_loss = total_loss / total_batches
    #     return avg_loss

    # Compute Metrics
    print("Model Size Comparison:")
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    print(f"Original Model Size: {original_size:.2f} MB")
    print(f"Quantized Model Size: {quantized_size:.2f} MB")
    print(f"Size Reduction: {(1 - quantized_size/original_size)*100:.2f}%")

    print("\nInference Time Comparison:")
    original_inference_time = measure_inference_time(original_model, test_loader)
    quantized_inference_time = measure_inference_time(quantized_model, test_loader)
    print(f"Original Model Inference Time: {original_inference_time*1000:.2f} ms")
    print(f"Quantized Model Inference Time: {quantized_inference_time*1000:.2f} ms")
    print(f"Inference Time Reduction: {(1 - quantized_inference_time/original_inference_time)*100:.2f}%")

    # print("\nLoss Comparison:")
    # original_loss = compute_loss(original_model, test_loader)
    # quantized_loss = compute_loss(quantized_model, test_loader)
    # print(f"Original Model Loss: {original_loss:.4f}")
    # print(f"Quantized Model Loss: {quantized_loss:.4f}")
    # print(f"Loss Difference: {abs(original_loss - quantized_loss):.4f}")

    return {
        'original_size': original_size,
        'quantized_size': quantized_size,
        'original_inference_time': original_inference_time,
        'quantized_inference_time': quantized_inference_time,
        # 'original_loss': original_loss,
        # 'quantized_loss': quantized_loss
    }

# Run the comparison
metrics = compute_model_metrics(model, loaded_model, test_loader)

# %%
