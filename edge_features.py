# edge\_features.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

__all__ = [
    "compute_enhanced_edge_features",
    "EdgeAugmentedConv",
]

@torch.no_grad()
def compute_enhanced_edge_features(data, eps: float = 1e-8):
    """
    Build edge_attr once (to be cached inside each Data object).
    Expected attributes on `data`:
      - data.pos: [N, 3]
      - data.edge_index: [2, E]
      - (optional) data.normals: [N, 3]
    Returns:
      - edge_attr: [E, D] where D = 3(rel) + 1(dist) + 3(unit) + 1(optional cos) = 7 or 8
    """
    assert hasattr(data, 'pos') and hasattr(data, 'edge_index'), "data.pos and data.edge_index required"
    pos = data.pos
    ei = data.edge_index
    row, col = ei[0], ei[1]

    # relative geometry
    rel = pos[col] - pos[row]                 # [E, 3]
    dist = torch.norm(rel, dim=1, keepdim=True).clamp_min(eps)  # [E, 1]
    rel_unit = rel / dist                     # [E, 3]

    feats = [rel, dist, rel_unit]

    # optional normal interaction
    nrm = getattr(data, 'normals', None)
    if nrm is not None:
        n_i = F.normalize(nrm[row], dim=1, eps=eps)
        cos_n_rel = (n_i * rel_unit).sum(dim=1, keepdim=True)  # [E, 1]
        feats.append(cos_n_rel)

    edge_attr = torch.cat(feats, dim=1)       # [E, D]
    return edge_attr


class EdgeAugmentedConv(nn.Module):
    """
    GCNConv + edge_attr-driven scalar gating.
    - Learns alpha(edge_attr) in [0,1] and multiplies into edge_weight.
    - Keeps shapes simple & fast. Works with cached edge_attr and optional edge_weight.
    """
    def __init__(self, in_ch: int, out_ch: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch, add_self_loops=False, normalize=True)
        hidden = max(16, min(128, edge_dim * 2))
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # alpha in [0,1]
        )
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.residual = (in_ch == out_ch)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        if edge_attr is None:
            # fall back: behave like plain GCNConv
            h = self.conv(x, edge_index, edge_weight=edge_weight)
        else:
            alpha = self.edge_mlp(edge_attr).squeeze(-1)  # [E]
            if edge_weight is None:
                ew = alpha
            else:
                # combine with precomputed weights
                ew = edge_weight * alpha
            h = self.conv(x, edge_index, edge_weight=ew)
        h = self.norm(h)
        h = F.silu(h, inplace=True)
        h = self.dropout(h)
        if self.residual:
            h = h + x
        return h


