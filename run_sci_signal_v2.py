import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# --- CONFIGURATION v3 ---
BATCH_SIZE = 64
TRAIN_SIZE = 4000      # Increased for stability
TEST_SIZE = 1000
EPOCHS = 5             # Increased for better convergence
SP_TARGET = 0.85       # Realistically calibrated target (was 0.95)
MAX_STEPS = 15         # Give controller room to work
TEMPERATURE = 0.5      # Temperature scaling (sharpening)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL DEFINITION ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # Larger filters
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x # Returns logits

# --- 2. UTILS: SP CALCULATION ---
def compute_sp(probs):
    """SP = 1 - (Entropy / MaxEntropy)"""
    probs = torch.clamp(probs, min=1e-9)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    max_entropy = np.log(10)
    sp = 1.0 - (entropy / max_entropy)
    return sp

# --- 3. TRAINING ---
def train_model():
    print(f"Loading MNIST (Train: {TRAIN_SIZE}, Test: {TEST_SIZE})...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(Subset(full_train, range(TRAIN_SIZE)), batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model

# --- 4. EVALUATION ---
def evaluate(model):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = DataLoader(Subset(datasets.MNIST('./data', train=False, transform=transform), range(TEST_SIZE)), batch_size=1, shuffle=False)
    
    base_acc, sci_acc = 0, 0
    base_sp_list, sci_sp_list = [], []
    sci_steps_list = []
    
    model.train() # Stochastic mode ON
    
    print(f"Running Inference (Target SP={SP_TARGET}, Temp={TEMPERATURE})...")
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # --- BASELINE (1 Pass) ---
            logits = model(data)
            # Apply Temperature Scaling to Baseline too for fair comparison
            probs = F.softmax(logits / TEMPERATURE, dim=1) 
            sp = compute_sp(probs)
            pred = probs.argmax(dim=1)
            
            base_acc += pred.eq(target).sum().item()
            base_sp_list.append(sp.item())
            
            # --- SCI (Logit Averaging Controller) ---
            accum_logits = logits.clone() # Start with first pass logits
            steps = 1
            current_sp = sp.item()
            
            # Loop: while quality is low, compute more
            while current_sp < SP_TARGET and steps < MAX_STEPS:
                new_logits = model(data)
                accum_logits += new_logits
                steps += 1
                
                # KEY CHANGE: Average Logits -> Softmax (Not Average Probs)
                mean_logits = accum_logits / steps
                current_probs = F.softmax(mean_logits / TEMPERATURE, dim=1)
                current_sp = compute_sp(current_probs).item()
            
            # Final Decision
            final_mean_logits = accum_logits / steps
            sci_probs = F.softmax(final_mean_logits / TEMPERATURE, dim=1)
            sci_pred = sci_probs.argmax(dim=1)
            
            sci_acc += sci_pred.eq(target).sum().item()
            sci_sp_list.append(current_sp)
            sci_steps_list.append(steps)

    # --- 5. STATS ---
    base_acc_pct = 100.0 * base_acc / TEST_SIZE
    sci_acc_pct = 100.0 * sci_acc / TEST_SIZE
    mean_base_sp = np.mean(base_sp_list)
    mean_sci_sp = np.mean(sci_sp_list)
    
    base_errors = [abs(SP_TARGET - sp) for sp in base_sp_list]
    sci_errors = [abs(SP_TARGET - sp) for sp in sci_sp_list]
    
    mean_base_error = np.mean(base_errors)
    mean_sci_error = np.mean(sci_errors)
    reduction = (mean_base_error - mean_sci_error) / mean_base_error * 100.0
    avg_steps = np.mean(sci_steps_list)

    print("\n" + "="*65)
    print(f"RESULTS v3: SCI (Logit Avg + Temp Scaling) vs Baseline")
    print("="*65)
    print(f"{'Metric':<25} | {'Baseline':<10} | {'SCI (Adaptive)':<15}")
    print("-" * 65)
    print(f"{'Accuracy':<25} | {base_acc_pct:.2f}%     | {sci_acc_pct:.2f}%")
    print(f"{'Mean Surgical Precision':<25} | {mean_base_sp:.4f}     | {mean_sci_sp:.4f}")
    print(f"{'Mean Steps':<25} | {1.0:.2f}       | {avg_steps:.2f}")
    print("-" * 65)
    print(f"{'Interpretive Error (dSP)':<25} | {mean_base_error:.4f}     | {mean_sci_error:.4f}")
    print(f"{'Error Reduction':<25} | -          | {reduction:.2f}%")
    print("="*65)

if __name__ == "__main__":
    trained_model = train_model()
    evaluate(trained_model)