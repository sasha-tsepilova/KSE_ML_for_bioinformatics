import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, multilabel_confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

TARGET_COLUMNS = [
    "Cytoplasm", "Nucleus", "Cell membrane", "Mitochondrion",
    "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"
]

def load_splits(data_path: Path, hpa_path: Path):
    swiss = pd.read_csv(data_path).rename(
        columns={"ACC": "protein_id", "Sequence": "sequence", "Partition": "partition"}
    )
    hpa = pd.read_csv(hpa_path).rename(columns={"sid": "protein_id", "fasta": "sequence"})

    val = swiss[swiss["partition"] == 4].copy()
    val = val[val[TARGET_COLUMNS].sum(axis=1) > 0].reset_index(drop=True)
    test = hpa.reset_index(drop=True)
    return val, test

class ProteinDataset(Dataset):
    def __init__(self, frame, tokenizer, label_columns, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["sequence"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(row[self.label_columns].to_numpy(dtype=np.float32), dtype=torch.float32),
        }

def masked_mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

class ProteinFoundationClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim, n_labels):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean_pool(output.last_hidden_state, attention_mask)
        return self.classifier(pooled)

@torch.no_grad()
def collect_probabilities(model, loader, device):
    model.eval()
    all_true, all_probs = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().numpy()
        
        logits = model(input_ids, attention_mask)
        probs = (1.0 / (1.0 + torch.exp(-logits))).cpu().numpy()
        
        all_true.append(labels)
        all_probs.append(probs)
        
    return np.vstack(all_true), np.vstack(all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--hpa-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing best_model.pt")
    parser.add_argument("--model-name", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--max-length", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_df, test_df = load_splits(args.data_path, args.hpa_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)
    
    model = ProteinFoundationClassifier(backbone, backbone.config.hidden_size, len(TARGET_COLUMNS)).to(device)
    
    checkpoint_path = args.model_dir / "best_model.pt"
    print(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    val_loader = DataLoader(ProteinDataset(val_df, tokenizer, TARGET_COLUMNS, args.max_length), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ProteinDataset(test_df, tokenizer, TARGET_COLUMNS, args.max_length), batch_size=args.batch_size, shuffle=False)

    print("Collecting probabilities...")
    y_val_true, val_probs = collect_probabilities(model, val_loader, device)
    y_test_true, test_probs = collect_probabilities(model, test_loader, device)

    thresholds = [0.20, 0.35, 0.50, 0.65]
    val_scores = {}

    print("\nValidation Macro-F1 by Threshold:")
    for t in thresholds:
        val_preds = (val_probs >= t).astype(int)
        score = f1_score(y_val_true, val_preds, average="macro", zero_division=0)
        val_scores[t] = score
        print(f"Threshold {t:.2f}: {score:.4f}")

    (args.model_dir / "threshold_val_scores.json").write_text(json.dumps(val_scores, indent=2))

    plt.figure(figsize=(6, 4))
    plt.plot(list(val_scores.keys()), list(val_scores.values()), marker='o')
    plt.title("Validation Macro-F1 vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Macro-F1")
    plt.grid(True)
    plt.savefig(args.model_dir / "val_macro_f1_vs_threshold.png")
    print(f"Saved plot to {args.model_dir / 'val_macro_f1_vs_threshold.png'}")

    best_t = max(val_scores, key=val_scores.get)
    print(f"\nSelected Threshold from Validation: {best_t:.2f}")

    test_preds = (test_probs >= best_t).astype(int)
    test_macro_f1 = f1_score(y_test_true, test_preds, average="macro", zero_division=0)
    print(f"HPA Test Macro-F1 at Threshold {best_t:.2f}: {test_macro_f1:.4f}")

    (args.model_dir / "best_threshold_test_results.json").write_text(json.dumps({
        "best_val_threshold": best_t,
        "hpa_test_macro_f1": test_macro_f1
    }, indent=2))

    matrices = multilabel_confusion_matrix(y_test_true, test_preds)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for label_name, matrix, ax in zip(TARGET_COLUMNS, matrices, axes.ravel()):
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['absent', 'present'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
        ax.set_title(label_name)

    fig.suptitle(f'HPA Test Confusion Matrices (Threshold {best_t:.2f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(args.model_dir / "hpa_test_confusion_matrices.png")
    print(f"Saved confusion matrices to {args.model_dir / 'hpa_test_confusion_matrices.png'}")

    rows = []
    for i in range(15): 
        true_names = [label for label, val in zip(TARGET_COLUMNS, y_test_true[i]) if val == 1]
        pred_names = [label for label, val in zip(TARGET_COLUMNS, test_preds[i]) if val == 1]
        
        rows.append({
            'protein_id': test_df.iloc[i]['protein_id'],
            'true_labels': ', '.join(true_names) or 'none',
            'predicted_labels': ', '.join(pred_names) or 'none',
            'highest_probability': round(float(test_probs[i].max()), 4)
        })

    prediction_table = pd.DataFrame(rows)
    prediction_table.to_csv(args.model_dir / "prediction_sample.csv", index=False)
    print(f"Saved sample predictions to {args.model_dir / 'prediction_sample.csv'}")

if __name__ == "__main__":
    main()