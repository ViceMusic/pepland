import os

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score)
from torch.backends import cudnn
from torch import nn
from torch.utils.data import DataLoader, random_split

from model.model import PharmHGT
from model.supervised import (PepLandClassifier, SupervisedMolGraphDataset,
                              supervised_collate)


CONFIG = {
    # Data. CSV files must contain columns named by smiles_col and label_col.
    "train_csv": "data/your_dataset/train.csv",
    "test_csv": "data/your_dataset/test.csv",
    "smiles_col": "smiles",
    "label_col": "label",
    "valid_ratio": 0.1,

    # Repeated runs. Each run uses base_seed + run_id for reproducible splitting.
    "num_runs": 5,
    "base_seed": 0,

    # Training.
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "num_workers": 0,
    "device": "cuda:0",
    "output_dir": "outputs/supervised",

    # Model.
    "pool": "avg",
    "dropout": 0.1,
    "fragment": "258",
    "hid_dim": 300,
    "num_layer": 5,
    "atom_dim": 42,
    "bond_dim": 14,
    "pharm_dim": 196,
    "reac_dim": 14,
    "act": "ReLU",
}


def get_device():
    device_name = CONFIG["device"]
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def fix_random_seed(random_seed, cuda_deterministic=True):
    import random

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def build_encoder(device):
    return PharmHGT(CONFIG["hid_dim"], CONFIG["act"], CONFIG["num_layer"],
                    CONFIG["atom_dim"], CONFIG["bond_dim"],
                    CONFIG["pharm_dim"], CONFIG["reac_dim"]).to(device)


def build_dataset(csv_path):
    return SupervisedMolGraphDataset(csv_path,
                                     smiles_col=CONFIG["smiles_col"],
                                     label_col=CONFIG["label_col"],
                                     frag=CONFIG["fragment"])


def make_train_valid_loaders(seed):
    dataset = build_dataset(CONFIG["train_csv"])
    valid_size = max(1, int(len(dataset) * CONFIG["valid_ratio"]))
    train_size = len(dataset) - valid_size
    if train_size <= 0:
        raise ValueError("Training set is too small for a 10% validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_set, valid_set = random_split(dataset,
                                        [train_size, valid_size],
                                        generator=generator)
    train_loader = DataLoader(train_set,
                              batch_size=CONFIG["batch_size"],
                              shuffle=True,
                              num_workers=CONFIG["num_workers"],
                              collate_fn=supervised_collate)
    valid_loader = DataLoader(valid_set,
                              batch_size=CONFIG["batch_size"],
                              shuffle=False,
                              num_workers=CONFIG["num_workers"],
                              collate_fn=supervised_collate)
    return train_loader, valid_loader


def make_test_loader():
    test_set = build_dataset(CONFIG["test_csv"])
    return DataLoader(test_set,
                      batch_size=CONFIG["batch_size"],
                      shuffle=False,
                      num_workers=CONFIG["num_workers"],
                      collate_fn=supervised_collate)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for graphs, labels in loader:
        graphs = graphs.to(device)
        labels = labels.float().view(-1, 1).to(device)

        logits = model(graphs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for graphs, labels in loader:
            graphs = graphs.to(device)
            labels = labels.float().view(-1, 1).to(device)

            logits = model(graphs)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits).view(-1)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size
            all_probs.append(probs.cpu())
            all_labels.append(labels.view(-1).cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": accuracy_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def run_once(run_id):
    seed = CONFIG["base_seed"] + run_id
    fix_random_seed(seed)
    device = get_device()

    train_loader, valid_loader = make_train_valid_loaders(seed)
    test_loader = make_test_loader()

    encoder = build_encoder(device)
    model = PepLandClassifier(encoder,
                              hid_dim=CONFIG["hid_dim"],
                              num_classes=1,
                              pool=CONFIG["pool"],
                              dropout=CONFIG["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    best_valid_f1 = -1.0
    best_path = os.path.join(CONFIG["output_dir"],
                             f"best_supervised_run_{run_id}.pt")

    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        valid_metrics = evaluate(model, valid_loader, criterion, device)
        print(
            f"run {run_id} epoch {epoch:03d} "
            f"train_loss {train_loss:.4f} "
            f"valid_acc {valid_metrics['acc']:.4f} "
            f"valid_auprc {valid_metrics['auprc']:.4f} "
            f"valid_f1 {valid_metrics['f1']:.4f}"
        )

        if valid_metrics["f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"run {run_id} test "
        f"acc {test_metrics['acc']:.4f} "
        f"auprc {test_metrics['auprc']:.4f} "
        f"f1 {test_metrics['f1']:.4f} "
        f"precision {test_metrics['precision']:.4f} "
        f"recall {test_metrics['recall']:.4f}"
    )
    return test_metrics


def main():
    results = [run_once(run_id) for run_id in range(CONFIG["num_runs"])]
    metric_names = ["acc", "auprc", "f1", "precision", "recall"]

    print("\nAverage over {} runs:".format(CONFIG["num_runs"]))
    for name in metric_names:
        values = np.array([result[name] for result in results], dtype=float)
        print(f"{name}: {values.mean():.4f} +/- {values.std():.4f}")


if __name__ == "__main__":
    main()
