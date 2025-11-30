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
from sklearn.metrics import roc_auc_score

# --- CONFIGURATION ---
SEEDS = [42, 100, 2024]
BATCH_SIZE = 64
EPOCHS = 10
SP_TARGET = 0.85
MAX_STEPS = 25
PATIENCE = 3
TEMPERATURE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET: PHYSICS-BASED BEARINGS ---
class SyntheticBearings(Dataset):
    """
    Simulates rotating machinery.
    Class 0: Healthy (Sine Wave + Noise)
    Class 1: Inner Race Fault (Impulses at specific frequencies)
    """
    def __init__(self, n_samples):
        self.data = []
        self.targets = []
        t = np.linspace(0, 1, 200) # 200 time steps (0.2s at 1kHz)
        
        for _ in range(n_samples):
            # Base Carrier (Shaft Rotation 30Hz)
            signal = 0.5 * np.sin(2 * np.pi * 30 * t) 
            label = 0
            
            # Fault Injection (50% chance)
            if np.random.rand() > 0.5:
                label = 1
                # Fault: High freq impulses (120Hz) decaying exponentially
                fault_sig = 0.8 * np.sin(2 * np.pi * 120 * t) * np.exp(-5*t)
                signal += fault_sig
            
            # Industrial Noise
            signal += np.random.normal(0, 0.4, size=len(t))
            
            self.data.append(torch.tensor(signal, dtype=torch.float32).unsqueeze(0))
            self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# --- MODEL ---
class BearingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * 47, 2) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- UTILS ---
def compute_sp(probs):
    probs = torch.clamp(probs, min=1e-9)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    sp = 1.0 - (entropy / np.log(2))
    return sp

# --- RUNNER ---
def run_experiment(seed):
    print(f"Running Bearings Seed {seed}...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_ds = SyntheticBearings(2000)
    test_ds = SyntheticBearings(500)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    model = BearingCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for _ in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            
    # Eval SCI
    logs = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            accum = model(x)
            steps = 1
            sp_hist = []
            
            while steps < MAX_STEPS:
                new_logits = model(x)
                accum += new_logits
                steps += 1
                
                curr_prob = F.softmax(accum/steps, dim=1)
                curr_sp = compute_sp(curr_prob).item()
                sp_hist.append(curr_sp)
                
                # Convergence Check
                if len(sp_hist) >= PATIENCE:
                    if abs(sp_hist[-1] - sp_hist[-PATIENCE]) < 0.005 and curr_sp > 0.8:
                        break
            
            final_prob = F.softmax(accum/steps, dim=1)
            pred = final_prob.argmax().item()
            correct = (pred == y.item())
            delta = abs(SP_TARGET - curr_sp)
            
            logs.append({
                "correct": int(correct),
                "delta": delta,
                "steps": steps
            })
            
    return logs

def analyze(logs):
    df = pd.DataFrame(logs)
    errors = 1 - df['correct']
    
    # Safety Analysis
    auc = roc_auc_score(errors, df['delta'])
    steps_correct = df[df['correct']==1]['steps'].mean()
    steps_wrong = df[df['correct']==0]['steps'].mean()
    
    print("\n" + "="*40)
    print("BEARINGS (INDUSTRIAL) RESULTS")
    print("="*40)
    print(f"Error Rate:      {errors.mean()*100:.2f}%")
    print(f"Safety AUROC:    {auc:.4f}")
    print("-" * 40)
    print("Metacognition (Avg Steps):")
    print(f"Correct:         {steps_correct:.2f}")
    print(f"Wrong:           {steps_wrong:.2f}")
    print("="*40)

if __name__ == "__main__":
    all_logs = []
    for s in SEEDS:
        all_logs.extend(run_experiment(s))
    analyze(all_logs)