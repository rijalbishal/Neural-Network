# cora_link_prediction.py
# Link prediction on Cora using a 2-layer GCN encoder + dot-product decoder (PyG)
# Outputs AUC and AP for val/test sets.

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------- Config -----------------
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

hidden_dim = 64
emb_dim = 32
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 200

# ----------------- Load dataset -----------------
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]
print(dataset)
print("Nodes:", data.num_nodes, "Node features:", data.num_node_features, "Classes:", dataset.num_classes)

# We need to split edges for link prediction (this modifies data in-place)
data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.10)
# train edges will be in data.train_pos_edge_index (only positive edges)
# negative samples can be taken via negative_sampling when needed

# ----------------- Model -----------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x   # embeddings Z (N x emb_dim)

def decode(z, edge_index):
    # edge_index: [2, E]
    src, dst = edge_index
    scores = (z[src] * z[dst]).sum(dim=1)  # dot product
    return scores  # raw scores (can use sigmoid + BCEWithLogits)

# ----------------- Setup -----------------
model = GCNEncoder(dataset.num_node_features, hidden_dim, emb_dim, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

x = data.x.to(device)
train_edge_index = data.train_pos_edge_index.to(device)
# For convolution we need the training graph adjacency: use train_pos edges (undirected)
# note train_test_split_edges keeps the graph for message passing using train_pos
edge_index_for_message_passing = data.train_pos_edge_index.to(device)

# ----------------- Evaluation helpers -----------------
@torch.no_grad()
def get_link_scores(z, pos_edge_index, neg_edge_index):
    pos_scores = torch.sigmoid(decode(z, pos_edge_index).cpu()).numpy()
    neg_scores = torch.sigmoid(decode(z, neg_edge_index).cpu()).numpy()
    y_true = np.hstack([np.ones(pos_scores.shape[0]), np.zeros(neg_scores.shape[0])])
    y_scores = np.hstack([pos_scores, neg_scores])
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return auc, ap

# ----------------- Training -----------------
best_val_auc = 0.0
best_test_auc_at_val = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    z = model(x, edge_index_for_message_passing)  # embeddings using train graph
    # positive train edges
    pos_train_edge_index = data.train_pos_edge_index.to(device)
    # sample negative train edges (same number as positives)
    neg_train_edge_index = negative_sampling(
        edge_index=pos_train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_train_edge_index.size(1),
        method='sparse')

    # scores
    pos_scores = decode(z, pos_train_edge_index)
    neg_scores = decode(z, neg_train_edge_index)

    # labels
    pos_labels = torch.ones(pos_scores.size(0), device=device)
    neg_labels = torch.zeros(neg_scores.size(0), device=device)

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    loss = F.binary_cross_entropy_with_logits(scores, labels)

    loss.backward()
    optimizer.step()

    # Evaluate on val / test using the embeddings computed from the train graph (common practice)
    model.eval()
    with torch.no_grad():
        z_eval = model(x, edge_index_for_message_passing)
        # val pos/neg
        val_pos = data.val_pos_edge_index.to(device)
        val_neg = data.val_neg_edge_index.to(device)
        test_pos = data.test_pos_edge_index.to(device)
        test_neg = data.test_neg_edge_index.to(device)

        val_auc, val_ap = get_link_scores(z_eval, val_pos, val_neg)
        test_auc, test_ap = get_link_scores(z_eval, test_pos, test_neg)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_test_auc_at_val = test_auc
        # optionally save checkpoint
        torch.save(model.state_dict(), "best_model_linkpred.pt")

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f} AP {val_ap:.4f} | Test AUC {test_auc:.4f} AP {test_ap:.4f}")

print("Training finished.")
print(f"Best validation AUC: {best_val_auc:.4f}")
print(f"Test AUC at best validation epoch: {best_test_auc_at_val:.4f}")

# ---------- Plot ROC and Precision–Recall curves ----------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# reload best model
model.load_state_dict(torch.load("best_model_linkpred.pt"))
model.eval()
with torch.no_grad():
    z_final = model(x, edge_index_for_message_passing)
    test_pos = data.test_pos_edge_index.to(device)
    test_neg = data.test_neg_edge_index.to(device)

    pos_scores = torch.sigmoid(decode(z_final, test_pos)).cpu().numpy()
    neg_scores = torch.sigmoid(decode(z_final, test_neg)).cpu().numpy()
    y_true = np.hstack([np.ones(pos_scores.shape[0]), np.zeros(neg_scores.shape[0])])
    y_scores = np.hstack([pos_scores, neg_scores])

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {best_test_auc_at_val:.4f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Link Prediction on Cora")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("cora_linkpred_ROC.png", dpi=300)
plt.show()

# Precision–Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"AP = {average_precision_score(y_true, y_scores):.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Link Prediction on Cora")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("cora_linkpred_PR.png", dpi=300)
plt.show()

print("Saved plots: cora_linkpred_ROC.png and cora_linkpred_PR.png")

