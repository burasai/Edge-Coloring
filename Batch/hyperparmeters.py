import torch

device = torch.device("cpu")
# max_colors = 30
max_colors = 7 # total colors max_colors +1 (including 0)
num_actions = max_colors + 1
in_feats = num_actions + 1
hidden_dim = 16
num_gat_heads = 4
gamma=0.99
lam=0.95
