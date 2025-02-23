import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_gat_heads, num_actions):
        """
        ActorCritic network using GAT for node embeddings.
        
        Parameters:
            in_feats (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden node embeddings.
            num_gat_heads (int): Number of attention heads in the GAT layer.
            num_actions (int): Number of possible edge colors (action space size).
        """
        super(ActorCritic, self).__init__()
        # GATConv: input dim -> hidden_dim * num_heads (we will average or concatenate)
        self.gat1 = GATConv(in_feats, hidden_dim, num_gat_heads, feat_drop=0.1, attn_drop=0.1, activation=F.elu)
        
        # We can combine multi-head outputs by concatenation.
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim * num_gat_heads * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)  # Output logits for each edge action
        )
        
        # Critic head: aggregate node embeddings to form a graph representation.
        # For aggregation, we use mean pooling.
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim * num_gat_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Value estimate
        )
        
    def forward(self, graph, node_features):
        """
        Forward pass for the ActorCritic network.
        
        Parameters:
            graph (DGLGraph): Input graph.
            node_features (Tensor): Node features tensor of shape (num_nodes, in_feats)
        
        Returns:
            edge_logits (Tensor): Logits for each edge (shape: (num_edges, num_actions))
            state_value (Tensor): Scalar value estimate for the graph (shape: (1,))
        """
        # Obtain node embeddings via GAT.
        # Output shape: (num_nodes, num_heads, hidden_dim). We then flatten heads.
        h = self.gat1(graph, node_features)  # shape: (N, num_heads, hidden_dim)
        N, num_heads, hidden_dim = h.shape
        h = h.reshape(N, num_heads * hidden_dim)  # shape: (N, hidden_dim * num_heads)
        
        # For the actor, we need edge-level features.
        # We can use DGL's apply_edges to compute the concatenation of source and destination node embeddings.
        graph.ndata['h'] = h  # store node embeddings
        def edge_feat_func(edges):
            # Concatenate source and destination embeddings.
            return {'edge_feat': torch.cat([edges.src['h'], edges.dst['h']], dim=1)}
        
        graph.apply_edges(edge_feat_func)
        edge_feat = graph.edata['edge_feat']  # shape: (num_edges, 2 * hidden_dim * num_heads)
        edge_logits = self.actor_fc(edge_feat)  # shape: (num_edges, num_actions)
        
        # For the critic, aggregate node embeddings to get a graph representation.
        # We use mean pooling.
        graph_rep = dgl.mean_nodes(graph, 'h')  # shape: (1, hidden_dim * num_heads)
        state_value = self.critic_fc(graph_rep)  # shape: (1, 1)
        state_value = state_value.squeeze()  # shape: scalar
        
        return edge_logits, state_value








def get_action(graph, node_features, actor_critic, device):
    """
    Given a graph and its node features, compute the edge action distribution using the actor,
    sample actions, and then enforce bidirectional consistency:
      For each edge (u,v) and its reverse (v,u), set both actions to the minimum of the two.
      
    Returns:
        action (Tensor): A tensor of shape (num_edges,) with the consistent actions.
        log_prob (Tensor): Log probabilities of the chosen actions.
        state_value (Tensor): Estimated state value (scalar).
    """
    actor_critic.eval()
    node_features = node_features.to(device)
    graph = graph.to(device)

    with torch.no_grad():
        edge_logits, state_value = actor_critic(graph, node_features)
        # Create a categorical distribution over edge actions
        dist = Categorical(logits=edge_logits)
        action = dist.sample()  # shape: (num_edges,)
        log_prob = dist.log_prob(action)
    
    # Enforce consistency on bidirectional edges:
    edges_src, edges_dst = graph.edges()
    num_edges = graph.num_edges()
    
    for i in range(num_edges):
        u = edges_src[i].item()
        v = edges_dst[i].item()
        # Find reverse edge: where src == v and dst == u
        reverse_indices = ((edges_src == v) & (edges_dst == u)).nonzero(as_tuple=True)[0]
        if len(reverse_indices) > 0:
            j = reverse_indices[0].item()
            # Take the minimum of the two actions
            a_i = action[i].item()
            a_j = action[j].item()
            new_val = min(a_i, a_j)
            action[i] = new_val
            action[j] = new_val

    return action, log_prob, state_value




