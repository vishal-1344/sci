import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
import random

# --- CONFIGURATION (COMPUTE MATCHED BASELINE) ---
# We match the ~15.6 steps from SCI v10
FIXED_K = 16  
SEEDS = [42, 100, 2024]
BATCH_SIZE = 64
EPOCHS = 10
TEMPERATURE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "experiments/mitbih_fixed_k"

# --- UTILS ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def compute_sp(probs):
    probs = torch.clamp(probs, min=1e-9)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    max_entropy = np.log(2)
    sp = 1.0 - (entropy / max_entropy)
    return sp

# --- DATASET ---
class RealMITBIH(Dataset):
    def __init__(self, csv_file, limit=None):
        df = pd.read_csv(csv_file, header=None)
        df.iloc[:, 187] = df.iloc[:, 187].apply(lambda x: 0 if x == 0 else 1)
        if limit:
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        self.y = df.iloc[:, 187].values.astype(int)
        self.X = df.iloc[:, :187].values.astype(np.float32)
        self.X = np.expand_dims(self.X, axis=1)
        num_neg = (self.y == 0).sum()
        num_pos = (self.y == 1).sum()
        self.pos_weight = num_neg / (num_pos + 1e-6)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# --- MODEL ---
class ECGCNN(nn.Module):
    def __init__(self):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# --- RUNNER ---
def run_experiment(seed):
    print(f"\n>>> Running Fixed-K Baseline (K={FIXED_K}), Seed {seed}...")
    set_seed(seed)
    
    train_ds = RealMITBIH("mitbih_train.csv", limit=12000)
    test_ds = RealMITBIH("mitbih_test.csv", limit=2000)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = ECGCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    weight = torch.tensor([1.0, train_ds.pos_weight], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    model.train()
    for epoch in range(EPOCHS):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    per_example = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # FIXED K ENSEMBLE
            accum_logits = model(data)
            
            # Already did 1, do K-1 more
            for _ in range(FIXED_K - 1):
                accum_logits += model(data)
                
            final_mean_logits = accum_logits / FIXED_K
            probs = F.softmax(final_mean_logits / TEMPERATURE, dim=1)
            sp = compute_sp(probs).item()
            pred = probs.argmax(dim=1).item()
            correct = (pred == target.item())
            
            per_example.append({
                "seed": seed,
                "y_true": target.item(),
                "correct": bool(correct),
                "sp": sp,
                "steps": FIXED_K
            })

    # Basic stats for print
    acc = np.mean([1 if x['correct'] else 0 for x in per_example])
    return {"acc": acc}, per_example

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    all_metrics = []
    all_examples = []
    
    for seed in SEEDS:
        m, ex = run_experiment(seed)
        all_metrics.append(m)
        all_examples.extend(ex)
        print(f"Seed {seed} Fixed-K Accuracy: {m['acc']:.4f}")
        
    with open(f"{OUT_DIR}/per_example.jsonl", "w") as f:
        for e in all_examples:
            f.write(json.dumps(e) + "\n")
            
    print(f"\nDone. Logs saved to {OUT_DIR}")

if __name__ == "__main__":
    main()