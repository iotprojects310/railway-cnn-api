import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

from .data import load_safe_csv, window_data, generate_anomalies, AccelerometerDataset
from .model import Conv1DClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def collate_fn(batch):
    xs = [torch.tensor(b[0]) for b in batch]
    ys = torch.tensor([b[1] for b in batch])
    xs = torch.stack(xs, dim=0)
    return xs, ys


def train(args):
    data = load_safe_csv(args.csv)
    windows = window_data(data, args.window_size, args.step)
    X, y = generate_anomalies(windows, seed=args.seed, anomaly_rate=args.anomaly_rate)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)

    train_ds = AccelerometerDataset(X_train, y_train)
    val_ds = AccelerometerDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv1DClassifier(in_channels=3, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train E{epoch}"):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                p = logits.argmax(dim=1).cpu().numpy()
                preds.extend(p.tolist())
                trues.extend(yb.numpy().tolist())
        acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state': model.state_dict()}, args.save)

    print('Best val acc:', best_acc)
    print('Classification report on validation:')
    print(classification_report(trues, preds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='safe_data.csv')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--step', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', type=str, default='models/best_model.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--anomaly_rate', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    train(args)
