"""
Prepare the Chinese Four Classics dataset for character-level language modeling.
Merges all four classic novels and creates character-level encoding.
"""
import os
import pickle
import numpy as np

# Get the directory of this script
data_dir = os.path.dirname(__file__)

# List of the four classic novels
novels = [
    'hongloumeng.txt',    # 红楼梦
    'sanguoyanyi.txt',    # 三国演义
    'shuihuzhuan.txt',    # 水浒传
    'xiyouji.txt',        # 西游记
]

# Read and merge all novels
print("Reading and merging Chinese classic novels...")
all_text = []
for novel in novels:
    novel_path = os.path.join(data_dir, novel)
    if os.path.exists(novel_path):
        with open(novel_path, 'r', encoding='utf-8') as f:
            text = f.read()
            all_text.append(text)
            print(f"  Loaded {novel}: {len(text):,} characters")
    else:
        print(f"  Warning: {novel} not found, skipping...")

# Merge all text
data = ''.join(all_text)
print(f"\nTotal length of dataset in characters: {len(data):,}")

# Get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Unique characters: {vocab_size:,}")
print(f"Sample characters: {''.join(chars[:])}...")

# Create mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

def decode(l):
    return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Create train and validation splits (90% train, 10% val)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"\ntrain has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# Save meta information
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nData preparation complete!")
print(f"Files saved:")
print(f"  - {os.path.join(data_dir, 'train.bin')}")
print(f"  - {os.path.join(data_dir, 'val.bin')}")
print(f"  - {os.path.join(data_dir, 'meta.pkl')}")

