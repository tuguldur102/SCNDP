import os
from model import SAGEEdgeProbModel
from dataloader import GraphEPCDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics.classification import BinaryAUROC
import torch.nn as nn, torch

GRAPHS_DIR_CLA = f"data/sparse/instances"
LABELS_DIR_CLA = f"data/sparse/labels"

# Optuna find those values for the best performance
BATCH = 128
EPOCHS = 30
LR = 0.00151785632
AGGR = 'mean' 
INPUT_DIM = 11
HID_DIM = 256
NUM_HEADS = 8
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SAGEEdgeProbModel(
  in_dim=INPUT_DIM, hidden_dim=HID_DIM, heads=NUM_HEADS, dropout=0.4, aggr=AGGR).to(device)

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)

pos_weight = torch.tensor(9.0, device=device)
loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_ds = GraphEPCDataset(GRAPHS_DIR_CLA, LABELS_DIR_CLA, 'train', AGGR)
val_ds   = GraphEPCDataset(GRAPHS_DIR_CLA, LABELS_DIR_CLA, 'val', AGGR)

print(f"# graphs in training set : {len(train_ds)}")
print(f"# graphs in validation set: {len(val_ds)}")

train_loader = PyGDataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = PyGDataLoader(val_ds,   batch_size=BATCH)

best_val = float('inf')
for epoch in range(1, EPOCHS + 1):
  model.train()
  total_loss = 0.0

  for batch in train_loader:
    batch = batch.to(device)
    logits = model(batch.x, batch.edge_index, batch.edge_prob)
    loss   = loss_fn(logits, batch.y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    total_loss += loss.item()

  avg_loss = total_loss / len(train_loader)

  # validation
  model.eval()
  val_loss = 0.0
  auroc = BinaryAUROC().to(device)

  with torch.no_grad():
    for batch in val_loader:
      batch = batch.to(device)
      logits = model(batch.x, batch.edge_index, batch.edge_prob)
      auroc.update(logits, batch.y.int())
      val_loss += loss_fn(logits, batch.y).item()

  val_loss /= len(val_loader)
  scheduler.step(val_loss)

  print(f"Epoch {epoch:02d} | train {avg_loss:.4f} "
  f"| val {val_loss:.4f} | AUROC {auroc.compute():.4f}")

  if val_loss < best_val:
    best_val = val_loss
    base_dir = "/checkpoints"

  torch.save(model.state_dict(),os.path.join(base_dir, f'best_model_cla_{EPOCHS}.pt'))
