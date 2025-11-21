# citeseer_gcn_fixed.py
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load Citeseer
dataset = Planetoid(root='data/Planetoid', name='Citeseer')
data = dataset[0].to(device)

print("Dataset:", dataset)
print("Nodes:", data.num_nodes)
print("Node features:", data.num_node_features)
print("Classes:", dataset.num_classes)
print("Edge index shape:", data.edge_index.shape)
print("Has train/val/test masks:", hasattr(data, 'train_mask'))

# Hyperparameters
in_dim = dataset.num_node_features
hidden_dim = 16
out_dim = dataset.num_classes
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 200

# Model
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(in_dim, hidden_dim, out_dim, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def accuracy(pred, labels):
    return (pred == labels).sum().item() / labels.size(0)

# ---------- Single training loop that records metrics ----------
loss_list = []
train_acc_list = []
val_acc_list = []
test_acc_list = []

best_val = 0.0
best_test = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)                  # forward
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()                                       # backward once per epoch
    optimizer.step()

    # evaluation (no gradient tracking)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        train_acc = accuracy(pred[data.train_mask], data.y[data.train_mask])
        val_acc = accuracy(pred[data.val_mask], data.y[data.val_mask])
        test_acc = accuracy(pred[data.test_mask], data.y[data.test_mask])

    # append metrics
    loss_list.append(loss.item())
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    test_acc_list.append(test_acc)

    # track best val/test
    if val_acc > best_val:
        best_val = val_acc
        best_test = test_acc

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
              f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")

print("Training finished.")
print(f"Best validation accuracy: {best_val:.4f}")
print(f"Test accuracy at best validation epoch: {best_test:.4f}")

# ---------- Improved plot (loss vs accuracy %) ----------
import numpy as np
val_pct = [v * 100.0 for v in val_acc_list]
test_pct = [t * 100.0 for t in test_acc_list]
epochs_range = np.arange(1, len(loss_list) + 1)

fig, ax1 = plt.subplots(figsize=(10,5))

# Left axis: Loss
lns1 = ax1.plot(epochs_range, loss_list, label='Train Loss', linewidth=1.2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Train Loss', fontsize=12)
ax1.tick_params(axis='y')
ax1.grid(True, linestyle='--', alpha=0.4)

# Right axis: Accuracy %
ax2 = ax1.twinx()
lns2 = ax2.plot(epochs_range, val_pct, label='Val Accuracy (%)', color='C1', linewidth=1.2, marker='o', markersize=4)
lns3 = ax2.plot(epochs_range, test_pct, label='Test Accuracy (%)', color='C2', linewidth=1.2, marker='s', markersize=4)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.tick_params(axis='y')

# Combined legend
lns = lns1 + lns2 + lns3
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='lower right', framealpha=0.9)

plt.title('GCN Training on Citeseer', fontsize=14)
fig.tight_layout()
outname = 'citeseer_training_plot_fixed.png'
plt.savefig(outname, dpi=300)
plt.show()

print(f"Saved improved plot to: {outname}")
