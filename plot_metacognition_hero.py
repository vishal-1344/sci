import json
import matplotlib.pyplot as plt
import numpy as np
import os

DOMAINS = [
    ("experiments/mnist_sci_v2", "MNIST (Vision)"),
    ("experiments/mitbih_sci_v2", "MIT-BIH (Medical)")
]

def plot():
    plt.figure(figsize=(10, 4))
    
    for i, (path, name) in enumerate(DOMAINS):
        log = os.path.join(path, "per_example.jsonl")
        if not os.path.exists(log): continue
        
        data = []
        with open(log, 'r') as f:
            for l in f: data.append(json.loads(l))
            
        corr = [d['steps'] for d in data if d['correct_sci']]
        wrong = [d['steps'] for d in data if not d['correct_sci']]
        
        plt.subplot(1, 2, i+1)
        plt.hist(corr, bins=np.arange(1, 26)-0.5, alpha=0.6, density=True, label='Correct', color='green')
        plt.hist(wrong, bins=np.arange(1, 26)-0.5, alpha=0.6, density=True, label='Incorrect', color='red')
        plt.title(f"{name}: Adaptive Compute")
        plt.xlabel("Inference Steps")
        plt.ylabel("Density")
        plt.legend()
        
    plt.tight_layout()
    plt.savefig("metacognition_hero.png")
    print("Saved metacognition_hero.png")

if __name__ == "__main__":
    plot()