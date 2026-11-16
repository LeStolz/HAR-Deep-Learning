# %% [markdown]
# # Part 4: Regularization and Robustness
#
# In Part 3, we observed that deeper networks achieve $>98.5\%$ training accuracy while validation accuracy plateaus, creating a generalization gap of $3.5\% - 4.5\%$.
# In this section, we apply and compare systematic regularization techniques:
# - **Dropout:** $p \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$
# - **Weight Decay ($L_2$ Regularization):** $\lambda \in \{0.0, 10^{-4}, 10^{-3}, 10^{-2}\}$ via AdamW
# - **Normalization:** Batch Normalization (`BatchNorm1d`), Layer Normalization (`LayerNorm`), and None
# - **Early Stopping:** Monitored on validation loss
#
# We perform a **Randomized Grid Search** across the top 3 architectures identified in Part 3:
# 1. `(GELU) x4` $[256, 128, 64, 32]$
# 2. `(Tanh) - (ReLU) x3` $[256, 128, 64, 32]$
# 3. `(ReLU) x4` $[256, 128, 64, 32]$

# %% [Imports & Setup]
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_seed, load_data, train_model, evaluate_model, plot_curves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_seed(42)

# Load Data
X_train, y_train, X_val, y_val, X_test, y_test, train_ds, val_ds, test_ds, feature_cols, target_col, idx_to_label = load_data()

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss()
label_names = [idx_to_label[i] for i in range(len(idx_to_label))]

# %% [RegularizedDeepMLP Architecture Definition]
class RegularizedDeepMLP(nn.Module):
    """
    Deep MLP supporting:
    - Arbitrary layer dimensions and activation sequences
    - Dropout rate
    - Normalization: 'none', 'batch_norm' (BatchNorm1d), or 'layer_norm' (LayerNorm)
    Layer sequence: Linear -> Normalization -> Activation -> Dropout
    """
    def __init__(self, input_dim=561, hidden_dims=[128, 64], num_classes=6,
                 activations="relu", dropout=0.0, norm_type="none"):
        super().__init__()
        if isinstance(activations, str):
            activations = [activations] * len(hidden_dims)

        layers = []
        prev_dim = input_dim

        for h_dim, act_name in zip(hidden_dims, activations):
            layers.append(nn.Linear(prev_dim, h_dim))

            if norm_type == "batch_norm":
                layers.append(nn.BatchNorm1d(h_dim))
            elif norm_type == "layer_norm":
                layers.append(nn.LayerNorm(h_dim))

            act = act_name.lower()
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "tanh":
                layers.append(nn.Tanh())
            elif act == "sigmoid":
                layers.append(nn.Sigmoid())
            elif act == "gelu":
                layers.append(nn.GELU())
            elif act == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            else:
                raise ValueError(f"Unsupported activation: {act_name}")

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# %% [Randomized Grid Search Setup & Execution]
top_architectures = [
    {"name": "(GELU) x4",          "dims": [256, 128, 64, 32], "acts": ["gelu", "gelu", "gelu", "gelu"]},
    {"name": "(Tanh) - (ReLU) x3", "dims": [256, 128, 64, 32], "acts": ["tanh", "relu", "relu", "relu"]},
    {"name": "(ReLU) x4",          "dims": [256, 128, 64, 32], "acts": ["relu", "relu", "relu", "relu"]},
]

param_grid = {
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
    "weight_decay": [0.0, 1e-4, 1e-3, 1e-2],
    "norm_type": ["none", "batch_norm", "layer_norm"],
}

# Fix random seed for reproducible trial selection
random.seed(42)
N_TRIALS = 15
search_trials = []

# Include unregularized baselines explicitly
for arch in top_architectures:
    search_trials.append({
        "trial": len(search_trials) + 1,
        "arch_name": arch["name"],
        "dims": arch["dims"],
        "acts": arch["acts"],
        "dropout": 0.0,
        "weight_decay": 0.0,
        "norm_type": "none"
    })

# Sample diverse combinations for the remaining trials
while len(search_trials) < N_TRIALS:
    arch = random.choice(top_architectures)
    drop = random.choice(param_grid["dropout"])
    wd = random.choice(param_grid["weight_decay"])
    norm = random.choice(param_grid["norm_type"])
    
    candidate = {
        "trial": len(search_trials) + 1,
        "arch_name": arch["name"],
        "dims": arch["dims"],
        "acts": arch["acts"],
        "dropout": drop,
        "weight_decay": wd,
        "norm_type": norm
    }
    search_trials.append(candidate)

print(f"Executing Randomized Grid Search ({len(search_trials)} trials)...")
random_search_results = []
trial_histories = {}
trained_models = {}

for t in search_trials:
    set_seed(42)
    model = RegularizedDeepMLP(
        hidden_dims=t["dims"], activations=t["acts"],
        dropout=t["dropout"], norm_type=t["norm_type"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=t["weight_decay"])

    t0 = time.perf_counter()
    hist = train_model(
        model, train_loader, val_loader, criterion, opt,
        n_epochs=35, device=device, early_stopping_patience=10, verbose_every=0
    )
    elapsed = time.perf_counter() - t0

    best_ep = hist["best_epoch"]
    n_eps = len(hist["train_loss"])
    time_per_ep = (elapsed / n_eps) * 1000

    train_acc = hist["train_acc"][best_ep - 1] * 100
    train_loss = hist["train_loss"][best_ep - 1]
    val_acc = hist["val_acc"][best_ep - 1] * 100
    val_loss = hist["val_loss"][best_ep - 1]
    val_f1 = hist["val_f1"][best_ep - 1] * 100
    gen_gap = train_acc - val_acc

    trial_name = f"Trial {t['trial']}: {t['arch_name']} [{t['norm_type']}]"
    trial_histories[trial_name] = hist
    trained_models[t["trial"]] = model

    random_search_results.append({
        "Trial": t["trial"],
        "Architecture": t["arch_name"],
        "Norm": t["norm_type"],
        "Dropout": t["dropout"],
        "Weight Decay": t["weight_decay"],
        "Best Epoch": f"{best_ep}/{n_eps}",
        "Train Acc (%)": round(train_acc, 2),
        "Train Loss": round(train_loss, 4),
        "Val Acc (%)": round(val_acc, 2),
        "Val Macro F1 (%)": round(val_f1, 2),
        "Val Loss": round(val_loss, 4),
        "Gen Gap (%)": round(gen_gap, 2),
        "Time/ep (ms)": round(time_per_ep, 1)
    })
    print(f"  Trial {t['trial']:>2} | {t['arch_name']:<20} | Norm: {t['norm_type']:<10} | Drop: {t['dropout']} | WD: {t['weight_decay']} -> Val Acc: {val_acc:.2f}% | Macro F1: {val_f1:.2f}% | Gap: {gen_gap:.2f}%")

df_search = pd.DataFrame(random_search_results).sort_values(by="Val Macro F1 (%)", ascending=False).reset_index(drop=True)
print("\n=== Randomized Grid Search Results (README 4.1) ===")
display(df_search)

# %% [Visualizations: Impact of Normalization, Dropout, and Weight Decay]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Validation Macro F1 by Normalization Type
sns.boxplot(
    data=df_search, x="Norm", y="Val Macro F1 (%)", hue="Norm",
    palette="Blues", ax=axes[0], legend=False, width=0.4, boxprops=dict(alpha=0.7)
)
sns.stripplot(data=df_search, x="Norm", y="Val Macro F1 (%)", ax=axes[0], color="black", size=7, jitter=0.15)
axes[0].set_title("Validation Macro F1 by Normalization", fontweight="bold")
axes[0].set_xlabel("Normalization Method")
axes[0].set_ylabel("Validation Macro F1 (%)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# 2. Generalization Gap vs. Dropout Rate
sns.regplot(
    data=df_search, x="Dropout", y="Gen Gap (%)", ax=axes[1],
    scatter_kws={"s": 60, "color": "crimson", "alpha": 0.8}, line_kws={"color": "darkred", "lw": 2}
)
axes[1].set_title("Generalization Gap vs. Dropout Rate\n(Higher Dropout shrinks the gap)", fontweight="bold")
axes[1].set_xlabel("Dropout Probability (p)")
axes[1].set_ylabel("Generalization Gap (Train - Val) (%)")
axes[1].grid(True, linestyle="--", alpha=0.5)

# 3. Validation Macro F1 vs. Weight Decay
sns.boxplot(
    data=df_search, x="Weight Decay", y="Val Macro F1 (%)", hue="Weight Decay",
    palette="Oranges", ax=axes[2], legend=False, width=0.4, boxprops=dict(alpha=0.7)
)
sns.stripplot(data=df_search, x="Weight Decay", y="Val Macro F1 (%)", ax=axes[2], color="black", size=7, jitter=0.15)
axes[2].set_title("Validation Macro F1 by Weight Decay", fontweight="bold")
axes[2].set_xlabel(r"Weight Decay ($\lambda$)")
axes[2].set_ylabel("Validation Macro F1 (%)")
axes[2].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [Comparative Learning Curves: Top Regularized Models vs. Unregularized Baseline]
top_trial_id = df_search.iloc[0]["Trial"]
second_trial_id = df_search.iloc[1]["Trial"]
unreg_trial = df_search[(df_search["Norm"] == "none") & (df_search["Dropout"] == 0.0)].iloc[0]["Trial"]

models_to_compare = {
    f"Rank 1: Trial {top_trial_id}": trained_models[top_trial_id],
    f"Rank 2: Trial {second_trial_id}": trained_models[second_trial_id],
    f"Unregularized: Trial {unreg_trial}": trained_models[unreg_trial]
}

plot_hists = {
    f"Rank 1: Trial {top_trial_id} ({df_search.iloc[0]['Norm']})": trial_histories[list(trial_histories.keys())[top_trial_id - 1]],
    f"Rank 2: Trial {second_trial_id} ({df_search.iloc[1]['Norm']})": trial_histories[list(trial_histories.keys())[second_trial_id - 1]],
    "Unregularized Baseline": trial_histories[list(trial_histories.keys())[unreg_trial - 1]]
}

plot_curves("Regularization Effect: Top Models vs. Unregularized", plot_hists)

# %% [markdown]
# ### Answers to Questions in Section 4 (README 4.1 & 4.2):
#
# #### 1. Effet de chaque méthode sur la stabilité et la généralisation (4.2):
#
# * **Batch Normalization (BatchNorm1d):**
#   * **Impact on Generalization:** **Largest performance boost.** BatchNorm elevated validation accuracy from $\approx 94.8\%$ to **$97.44\%$** and Macro F1 to **$97.67\%$**.
#   * **Impact on Stability:** By standardizing pre-activations to zero mean and unit variance at each layer, BatchNorm eliminates vanishing/exploding gradients and smooths the loss landscape, enabling deep 4-layer networks to train stably.
#   * **Regularizing Mechanism:** The mini-batch stochastic noise in $\mu_B$ and $\sigma_B$ acts as continuous data augmentation in latent space, preventing neurons from relying on specific feature scales.
#
# * **Layer Normalization (LayerNorm):**
#   * **Impact:** Provides consistent improvements over unnormalized models ($\approx 95.8\% - 96.0\%$), but trails BatchNorm slightly ($97.4\%$).
#   * **Stability:** Highly stable and invariant to batch size, making it a reliable alternative when batch sizes are very small.
#
# * **Dropout ($p \in [0.1, 0.5]$):**
#   * **Impact on Generalization Gap:** Dropout is the most effective tool for controlling the generalization gap.
#     * Without dropout ($p = 0.0$): the gap reaches **$4.33\%$** (severe memorization).
#     * At $p = 0.2 - 0.3$: the gap shrinks to **$1.4\% - 1.8\%$** while maintaining peak validation scores.
#     * At $p = 0.5$: the gap is **completely erased ($0.02\%$)**, though setting $p$ too high can mildly underfit.
#   * **Mechanism:** Randomly deactivating units prevents co-adaptation, forcing independent feature representation.
#
# * **Weight Decay ($L_2$ Regularization in AdamW, $\lambda \in [10^{-4}, 10^{-2}]$):**
#   * **Impact:** Decoupled weight decay ($\lambda = 10^{-4}$ to $10^{-3}$) prevents weight explosion on the 561 correlated sensor features, encouraging flatter, more robust decision boundaries. Setting $\lambda \ge 0.01$ over-penalizes weights and degrades performance.
#
# * **Early Stopping (`patience=10` to `20`):**
#   * **Impact:** Halts training precisely when validation loss plateaus (typically between epochs 12 and 30), preventing the network from continuing into the memorization regime.
#
# ---
#
# #### 2. Les 2 ou 3 configurations retenues pour l'évaluation finale (README 4.2):
#
# For final test evaluation in Part 5, we keep the top 3 best-performing and most complementary configurations:
# 1. **Model 1 (Top Performance): `(GELU) x4 + BatchNorm + Weight Decay 1e-4`**
#    * Layout: $[256, 128, 64, 32]$ GELU
#    * Regularization: `BatchNorm1d`, `Dropout = 0.0`, `Weight Decay = 0.0001`
#    * Validation: **Val Acc: 97.44% | Val Macro F1: 97.67% | Val Loss: 0.0741**
#
# 2. **Model 2 (Best Regularized / Balanced): `(GELU) x4 + BatchNorm + Dropout 0.3`**
#    * Layout: $[256, 128, 64, 32]$ GELU
#    * Regularization: `BatchNorm1d`, `Dropout = 0.3`, `Weight Decay = 0.0`
#    * Validation: **Val Acc: 97.02% | Val Macro F1: 97.27% | Gen Gap: 1.47%**
#
# 3. **Model 3 (Best Pure ReLU): `(ReLU) x4 + BatchNorm + Dropout 0.3`**
#    * Layout: $[256, 128, 64, 32]$ ReLU
#    * Regularization: `BatchNorm1d`, `Dropout = 0.3`, `Weight Decay = 0.0`
#    * Validation: **Val Acc: 96.48% | Val Macro F1: 96.79% | Gen Gap: 1.76%**
#    *(Alternative: `(Tanh) - (ReLU) x3 + Dropout 0.5 + WD 1e-3` achieving **95.95% Acc, 96.28% F1** with nearly zero generalization gap: **0.71%**).*

# %% [Multi-Seed Robustness on the Selected Top 3 Configurations (README 2.3)]
SEEDS = [42, 123, 999, 1000, 2026]

selected_configs = [
    {
        "name": "Config 1: (GELU) x4 + BN + WD 1e-4",
        "dims": [256, 128, 64, 32], "acts": ["gelu"] * 4,
        "norm": "batch_norm", "drop": 0.0, "wd": 1e-4
    },
    {
        "name": "Config 2: (GELU) x4 + BN + Drop 0.3",
        "dims": [256, 128, 64, 32], "acts": ["gelu"] * 4,
        "norm": "batch_norm", "drop": 0.3, "wd": 0.0
    },
    {
        "name": "Config 3: (ReLU) x4 + BN + Drop 0.3",
        "dims": [256, 128, 64, 32], "acts": ["relu"] * 4,
        "norm": "batch_norm", "drop": 0.3, "wd": 0.0
    }
]

multiseed_records = []
print(f"Validating Top 3 Regularized Configurations across {len(SEEDS)} seeds...")

for cfg in selected_configs:
    for s in SEEDS:
        set_seed(s)
        model = RegularizedDeepMLP(
            hidden_dims=cfg["dims"], activations=cfg["acts"],
            dropout=cfg["drop"], norm_type=cfg["norm"]
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=cfg["wd"])

        hist = train_model(
            model, train_loader, val_loader, criterion, opt,
            n_epochs=35, device=device, early_stopping_patience=10, verbose_every=0
        )
        best_ep = hist["best_epoch"]
        v_acc = hist["val_acc"][best_ep - 1] * 100
        v_f1 = hist["val_f1"][best_ep - 1] * 100
        gap = (hist["train_acc"][best_ep - 1] * 100) - v_acc

        multiseed_records.append({
            "Configuration": cfg["name"],
            "Seed": s,
            "Best Epoch": best_ep,
            "Val Acc (%)": round(v_acc, 2),
            "Val Macro F1 (%)": round(v_f1, 2),
            "Gen Gap (%)": round(gap, 2)
        })
        print(f"  [{cfg['name']}] Seed {s:>4} -> Val Acc: {v_acc:.2f}% | Macro F1: {v_f1:.2f}%")

df_multiseed = pd.DataFrame(multiseed_records)

# Summary table across seeds
ms_summary = []
for name, group in df_multiseed.groupby("Configuration", sort=False):
    accs = group["Val Acc (%)"]
    f1s = group["Val Macro F1 (%)"]
    gaps = group["Gen Gap (%)"]
    ms_summary.append({
        "Configuration": name,
        "Val Accuracy (Mean ± Std) (%)": f"{accs.mean():.2f} ± {accs.std():.2f}",
        "Val Macro F1 (Mean ± Std) (%)": f"{f1s.mean():.2f} ± {f1s.std():.2f}",
        "Gen Gap (Mean) (%)": f"{gaps.mean():.2f}",
        "Min - Max Val Acc (%)": f"{accs.min():.2f}% - {accs.max():.2f}%",
        "_sort": f1s.mean()
    })

df_final_selected = pd.DataFrame(ms_summary).sort_values(by="_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
print("\n=== Final Top 3 Regularized Configurations (Multi-Seed Verification) ===")
display(df_final_selected)
