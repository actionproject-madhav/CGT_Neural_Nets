"""Generate plots from saved model histories."""
import os, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

RESULTS = "results"
PLOTS = os.path.join(RESULTS, "plots")
os.makedirs(PLOTS, exist_ok=True)

# Load histories
mlp_hists = []
for seed in [42, 123, 456, 789, 1024]:
    path = f"{RESULTS}/mlp_baseline/seed_{seed}/history.json"
    if os.path.exists(path):
        with open(path) as f:
            mlp_hists.append(json.load(f))

cgt_hists = []
for seed in [42, 123, 456, 789, 1024]:
    path = f"{RESULTS}/cgt_net/seed_{seed}/history.json"
    if os.path.exists(path):
        with open(path) as f:
            cgt_hists.append(json.load(f))

print(f"Loaded {len(mlp_hists)} MLP histories, {len(cgt_hists)} CGT-Net histories")

# Plot learning curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# MLP
for i, h in enumerate(mlp_hists):
    a = 0.3 if i > 0 else 1.0
    lw = 1.5 if i == 0 else 1.0
    ep = range(1, len(h["train_acc"]) + 1)
    ax1.plot(ep, h["train_acc"], 'b-', alpha=a, lw=lw)
    ax1.plot(ep, h["val_acc"], 'b--', alpha=a, lw=lw)
ax1.set_title("MLP Baseline", fontsize=14, fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.4, 1.02)
ax1.grid(alpha=0.3)
ax1.axhline(0.95, color='gray', linestyle=':', alpha=0.5)

# CGT-Net
for i, h in enumerate(cgt_hists):
    a = 0.3 if i > 0 else 1.0
    lw = 1.5 if i == 0 else 1.0
    ep = range(1, len(h["train_acc"]) + 1)
    ax2.plot(ep, h["train_acc"], 'r-', alpha=a, lw=lw)
    ax2.plot(ep, h["val_acc"], 'r--', alpha=a, lw=lw)
ax2.set_title("CGT-Decomposition Network", fontsize=14, fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylim(0.4, 1.02)
ax2.grid(alpha=0.3)
ax2.axhline(0.95, color='gray', linestyle=':', alpha=0.5)

from matplotlib.lines import Line2D
legend = [Line2D([0],[0], color='gray', lw=2, label='Train'),
          Line2D([0],[0], color='gray', lw=2, ls='--', label='Val')]
ax2.legend(handles=legend, loc='lower right')

plt.suptitle("Training Curves: MLP vs CGT-Net (solid=train, dashed=val)", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS}/learning_curves.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved {PLOTS}/learning_curves.png")

# Plot validation accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(mlp_hists))
mlp_vals = [max(h["val_acc"]) for h in mlp_hists]
cgt_vals = [max(h["val_acc"]) for h in cgt_hists]

ax.bar(x - 0.2, mlp_vals, 0.4, label='MLP Baseline', color='#4C72B0', edgecolor='white')
ax.bar(x + 0.2, cgt_vals[:len(mlp_hists)], 0.4, label='CGT-Net', color='#C44E52', edgecolor='white')

seeds = [42, 123, 456, 789, 1024][:len(mlp_hists)]
ax.set_xticks(x)
ax.set_xticklabels([f'Seed {s}' for s in seeds])
ax.set_ylabel('Best Validation Accuracy', fontsize=13)
ax.set_title('Peak Validation Accuracy: MLP vs CGT-Net', fontsize=15, fontweight='bold')
ax.set_ylim(0.5, 1.05)
ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

for i, (m, c) in enumerate(zip(mlp_vals, cgt_vals[:len(mlp_hists)])):
    ax.text(i - 0.2, m + 0.01, f'{m:.1%}', ha='center', fontsize=10)
    ax.text(i + 0.2, c + 0.01, f'{c:.1%}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{PLOTS}/val_accuracy_comparison.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved {PLOTS}/val_accuracy_comparison.png")

# Summary stats
print(f"\n{'='*60}")
print("TRAINING SUMMARY")
print(f"{'='*60}")
print(f"\nMLP Baseline ({len(mlp_hists)} seeds):")
print(f"  Val accuracy: {np.mean(mlp_vals):.1%} ± {np.std(mlp_vals):.1%}")
print(f"  Range: {min(mlp_vals):.1%} - {max(mlp_vals):.1%}")
print(f"\nCGT-Net ({len(cgt_hists)} seeds):")
print(f"  Val accuracy: {np.mean(cgt_vals):.1%} ± {np.std(cgt_vals):.1%}")
print(f"  Range: {min(cgt_vals):.1%} - {max(cgt_vals):.1%}")
print(f"\n{'='*60}")
print(f"✓ Plots saved to {PLOTS}/")
print(f"{'='*60}")
