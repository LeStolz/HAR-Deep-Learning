import os, random, json, copy, torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

PROCESSED_DIR = os.path.join("data", "processed")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data():
	meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
	with open(meta_path, "r", encoding="utf-8") as f:
		metadata = json.load(f)

	idx_to_label = {int(k): v for k, v in metadata["idx_to_label"].items()}

	train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
	val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
	test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

	target_col = "Activity"
	feature_cols = [c for c in train_df.columns if c != target_col]

	X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
	y_train = torch.tensor(train_df[target_col].values, dtype=torch.long)

	X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
	y_val = torch.tensor(val_df[target_col].values, dtype=torch.long)

	X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
	y_test = torch.tensor(test_df[target_col].values, dtype=torch.long)

	train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
	val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
	test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

	return (
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_dataset, val_dataset, test_dataset,
        feature_cols, target_col, idx_to_label
	)


def grad_norm(model):
    sq_norms = [(p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None]
    return torch.sqrt(torch.stack(sq_norms).sum()).item()


def train_epoch(model, loader, criterion, optimizer, device, clip_norm=None):
    model.train()
    running_loss = 0.0
    grad_norms = []
    correct = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        yhat = model(X)
        loss = criterion(yhat, y)
        loss.backward()
        grad_norms.append(grad_norm(model))

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = yhat.argmax(dim=1)
        correct += (preds == y).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc, grad_norms


def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            loss = criterion(yhat, y)

            running_loss += loss.item() * X.size(0)
            preds = yhat.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    total = len(all_targets)
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1, np.array(all_preds), np.array(all_targets)


def train_model(
    model, train_loader, val_loader, criterion, optimizer, n_epochs, device,
    early_stopping_patience=None, scheduler=None, verbose_every=10, clip_norm=None
):
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "grad_norms": []
    }
    best_v_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc, grad_norms = train_epoch(model, train_loader, criterion, optimizer, device, clip_norm)
        v_loss, v_acc, v_f1, _, _ = evaluate_model(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)
        history["grad_norms"] += grad_norms

        if v_loss < best_v_loss:
            best_v_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if verbose_every and (epoch % verbose_every == 0 or epoch == 1 or epoch == n_epochs):
            print(f"Epoch {epoch:02d}/{n_epochs:02d} | "
                  f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}% | "
                  f"Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%, Macro F1: {v_f1*100:.2f}%")

        if early_stopping_patience is not None and wait >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best val loss at epoch {best_epoch}.")
            break

        if not np.isfinite(tr_loss):
            print(f"Training loss is NaN or Inf at epoch {epoch}. Stopping training.")
            break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_v_loss
    if early_stopping_patience is not None and best_state is not None:
        model.load_state_dict(best_state)

    return history


def lr_range_test(model, loader, criterion, make_optimizer, lr_min, lr_max, device, beta=0.9):
    n = len(loader)
    q = (lr_max / lr_min) ** (1.0 / (n - 1))

    optimizer = make_optimizer(model.parameters(), lr_min)
    lr, avg, best = lr_min, 0.0, float("inf")

    lrs, losses = [], []
    model.train()
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        # Moving average with correction
        avg = beta * avg + (1 - beta) * loss.item()
        smoothed = avg / (1 - beta ** (i + 1))

        lrs.append(lr); losses.append(smoothed)
        best = min(best, smoothed)

        if not np.isfinite(smoothed) or smoothed > 4 * best:
            break

        lr = lr * q

    return lrs, losses


def plot_curves(title, hists, key_pairs=(("train_loss", "val_loss"), ("train_acc", "val_acc"))):
    if "train_loss" in hists:
        hists = {title: hists}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (ktr, kva) in zip(axes, key_pairs):
        for i, (name, h) in enumerate(hists.items()):
            ax.plot(h[ktr], ls="--", c=f"C{i}", alpha=0.7)
            ax.plot(h[kva], ls="-", c=f"C{i}", label=name)
        ax.set_xlabel("Epoch"); ax.set_ylabel(kva.replace("val_", "")); ax.legend(fontsize=8)
    axes[0].set_title(title + "  (dashed: train, solid: validation)")
    plt.tight_layout(); plt.show()