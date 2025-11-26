"""
Sample text from the trained Chinese Four Classics model
"""
from contextlib import nullcontext
import os
import pickle
import torch

from model import GPT, GPTConfig

# Configuration
init_from = 'resume'
out_dir = 'out-chinese'
start = '\n'  # Starting prompt
num_samples = 5  # Number of samples to generate
max_new_tokens = 500  # Maximum tokens to generate
temperature = 0.8  # Sampling temperature (lower = more deterministic)
top_k = 200  # Top-k sampling
seed = 1337

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else 'float32'
compile = False

# Load configurator if exists
if os.path.exists('configurator.py'):
    exec(open('configurator.py').read())

print(f"Using device: {device}")

# Set random seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Please train the model first using: python train.py config/train_chinese.py")
        exit(1)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconfig = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconfig)
    state_dict = checkpoint['model']
    
    # Remove compiled model prefix if exists
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint '{ckpt_path}'")
    print(f"  Model args: {checkpoint['model_args']}")
    if 'iter_num' in checkpoint:
        print(f"  Trained for {checkpoint['iter_num']} iterations")
    if 'best_val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

if compile:
    try:
        model = torch.compile(model)
        print("Model compiled")
    except Exception as e:
        print(f"Warning: Model compilation failed: {e}")
        print("Continuing without compilation...")

# Load tokenizer (character-level for Chinese)
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f"Loaded character-level tokenizer from {meta_path}")
    print(f"  Vocabulary size: {len(stoi)}")
else:
    print("Warning: meta.pkl not found, using fallback encoding")
    # Fallback: simple character encoding
    def encode(s):
        return [ord(c) for c in s]
    def decode(l):
        return ''.join([chr(i) for i in l])

# Prepare input
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

print(f"\nStarting generation with prompt: '{start[:50]}...'")
print(f"Generating {num_samples} samples...")
print("=" * 80)

# Generate samples
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode(y[0].tolist())
            print(f"\nSample {k+1}:")
            print("-" * 80)
            print(generated_text)
            print("=" * 80)

print("\nGeneration complete!")

