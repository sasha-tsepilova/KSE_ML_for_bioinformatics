import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

TARGET_COLUMNS = [
    "Cytoplasm", "Nucleus", "Cell membrane", "Mitochondrion",
    "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"
]

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
            row["sequence"], 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0), 
            "attention_mask": encoded["attention_mask"].squeeze(0), 
            "label": torch.tensor(row[self.label_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
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
        return self.classifier(masked_mean_pool(output.last_hidden_state, attention_mask))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hpa = pd.read_csv("data/hpa_testset.csv").rename(columns={"sid": "protein_id", "fasta": "sequence"})
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    backbone = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = ProteinFoundationClassifier(backbone, backbone.config.hidden_size, len(TARGET_COLUMNS)).to(device)
    
    model.load_state_dict(torch.load("outputs/esm35m_full/best_model.pt", map_location=device))
    model.eval()
    
    test_loader = DataLoader(ProteinDataset(hpa, tokenizer, TARGET_COLUMNS, max_length=400), batch_size=8)
    
    all_true, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_true.append(batch["label"].cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            
    y_true = np.vstack(all_true)
    y_probs = np.vstack(all_probs)
    
    with open("outputs/esm35m_full/best_threshold_test_results.json") as f:
        best_t = json.load(f)["best_val_threshold"]
        
    y_pred = (y_probs >= best_t).astype(int)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    df = pd.DataFrame({"Label": TARGET_COLUMNS, "Finetuned_F1": per_class_f1})
    df.to_csv("outputs/finetuned_per_class_f1.csv", index=False)
    print("Saved per-class metrics to outputs/finetuned_per_class_f1.csv")

if __name__ == "__main__":
    main()