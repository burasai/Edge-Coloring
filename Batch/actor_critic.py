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
        # GATConv: input dim -> hidden_dim * num_heads (we will concatenate outputs)
        self.gat1 = GATConv(in_feats, hidden_dim, num_gat_heads, feat_drop=0.1, attn_drop=0.1, activation=F.elu)
        
        # Actor head: processes edge-level features.
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim * num_gat_heads * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)  # Outputs logits for each edge action
        )
        
        # Critic head: aggregates node embeddings for a graph-level value.
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim * num_gat_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs a scalar value
        )
        
    def forward(self, graph, node_features):
        """
        Forward pass for the ActorCritic network.
        
        Parameters:
            graph (DGLGraph): Input graph (can be batched or single).
            node_features (Tensor): Node features tensor of shape (num_nodes, in_feats).
        
        Returns:
            edge_logits (Tensor): Logits for each edge (shape: (num_edges, num_actions)).
            state_value (Tensor): Scalar value estimate for the graph (or aggregated over the batch).
        """
        # Obtain node embeddings via GAT.
        h = self.gat1(graph, node_features)  # Shape: (N, num_heads, hidden_dim)
        N, num_heads, hidden_dim = h.shape
        h = h.reshape(N, num_heads * hidden_dim)  # Shape: (N, hidden_dim * num_heads)
        
        # Store node embeddings in graph for later use.
        graph.ndata['h'] = h
        
        # Define an edge function that concatenates source and destination node embeddings.
        def edge_feat_func(edges):
            return {'edge_feat': torch.cat([edges.src['h'], edges.dst['h']], dim=1)}
        
        graph.apply_edges(edge_feat_func)
        edge_feat = graph.edata['edge_feat']  # Shape: (num_edges, 2 * hidden_dim * num_heads)
        edge_logits = self.actor_fc(edge_feat)  # Shape: (num_edges, num_actions)
        
        # For the critic: aggregate node embeddings using mean pooling.
        graph_rep = dgl.mean_nodes(graph, 'h')  # For batched graphs, this returns per-graph features.
        state_value = self.critic_fc(graph_rep)  # Shape: (num_graphs, 1) for batched graphs.
        state_value = state_value.mean()  # Aggregate to a scalar if needed.
        
        return edge_logits, state_value

# def get_action(graph, node_features, actor_critic, device):
#     """
#     Given a graph (or batched graph) and its node features, computes the edge action distribution 
#     using the actor, samples actions, and enforces bidirectional consistency (each edge and its reverse 
#     are set to the minimum sampled action).
    
#     This function automatically handles batched graphs by unbatching and processing each subgraph.
    
#     Parameters:
#         graph (DGLGraph): Input graph (batched or single).
#         node_features (Tensor): Node features tensor.
#         actor_critic (ActorCritic): The actor–critic network.
#         device (torch.device): Device to run the computation on.
        
#     Returns:
#         action (Tensor): Tensor of shape (num_edges,) with the final actions.
#         log_prob (Tensor): Log probabilities of the chosen actions.
#         state_value (Tensor): Estimated state value (scalar).
#     """
#     actor_critic.eval()
#     node_features = node_features.to(device)
#     graph = graph.to(device)
    
#     # Try unbatching the graph; if it is not batched, unbatch returns a list with a single graph.
#     subgraphs = dgl.unbatch(graph)
#     if len(subgraphs) > 1:
#         actions_list, log_probs_list, state_values_list = [], [], []
#         # Split node_features according to subgraph node counts.
#         node_counts = [g.num_nodes() for g in subgraphs]
#         node_features_split = torch.split(node_features, node_counts, dim=0)
#         for subgraph, nf in zip(subgraphs, node_features_split):
#             a, lp, sv = get_action(subgraph, nf, actor_critic, device)
#             actions_list.append(a)
#             log_probs_list.append(lp)
#             state_values_list.append(sv)
#         # Concatenate actions and log probabilities across subgraphs.
#         action = torch.cat(actions_list)
#         log_prob = torch.cat(log_probs_list)
#         state_value = torch.stack(state_values_list).mean()  # Aggregate state values (e.g., by averaging)
#         return action, log_prob, state_value
    
#     # Single-graph branch:
#     with torch.no_grad():
#         edge_logits, state_value = actor_critic(graph, node_features)
#         dist = Categorical(logits=edge_logits)
#         action = dist.sample()  # Shape: (num_edges,)
#         log_prob = dist.log_prob(action)
    
#     # Enforce bidirectional consistency:
#     edges_src, edges_dst = graph.edges()
#     num_edges = graph.number_of_edges()
    
#     for i in range(num_edges):
#         u = edges_src[i].item()
#         v = edges_dst[i].item()
#         reverse_indices = ((edges_src == v) & (edges_dst == u)).nonzero(as_tuple=True)[0]
#         if len(reverse_indices) > 0:
#             j = reverse_indices[0].item()
#             new_val = min(action[i].item(), action[j].item())
#             action[i] = new_val
#             action[j] = new_val

#     return action, log_prob, state_value






def get_action(graph, node_features, actor_critic, device):
    """
    Given a graph (or batched graph) and its node features, compute the edge action distribution
    using the actor, sample actions, and enforce bidirectional consistency by setting both (u,v) and (v,u)
    to the minimum of their sampled actions.

    If the graph is batched, we unbatch and process each subgraph independently.
    If a subgraph has zero edges, we return an empty tensor for that subgraph.

    Parameters:
        graph (DGLGraph): Input graph (batched or single).
        node_features (Tensor): Node features tensor.
        actor_critic (ActorCritic): The actor–critic network.
        device (torch.device): Device to run computations on.

    Returns:
        action (Tensor): Tensor of shape (num_edges,) with final actions.
        log_prob (Tensor): Log probabilities for chosen actions.
        state_value (Tensor): Estimated state value (scalar).
    """
    actor_critic.eval()
    node_features = node_features.to(device)
    graph = graph.to(device)
    
    # Check if the graph is batched by unbatching it.
    subgraphs = dgl.unbatch(graph)
    if len(subgraphs) > 1:
        actions_list, log_probs_list, state_values_list = [], [], []
        # Split node_features based on subgraph node counts.
        node_counts = [g.num_nodes() for g in subgraphs]
        node_features_split = torch.split(node_features, node_counts, dim=0)
        for subgraph, nf in zip(subgraphs, node_features_split):
            # If this subgraph has no edges, return empty tensors.
            if subgraph.number_of_edges() == 0:
                actions_list.append(torch.tensor([], device=device, dtype=torch.int64))
                log_probs_list.append(torch.tensor([], device=device, dtype=torch.float32))
                state_values_list.append(torch.tensor(0.0, device=device, dtype=torch.float32))
            else:
                a, lp, sv = get_action(subgraph, nf, actor_critic, device)
                actions_list.append(a)
                log_probs_list.append(lp)
                state_values_list.append(sv)
        # Concatenate results.
        action = torch.cat(actions_list)
        log_prob = torch.cat(log_probs_list)
        state_value = torch.stack(state_values_list).mean()  # Average state value.
        return action, log_prob, state_value

    # Single-graph branch:
    if graph.number_of_edges() == 0:
        # Return empty tensors if no edges.
        return torch.tensor([], device=device, dtype=torch.int64), torch.tensor([], device=device, dtype=torch.float32), torch.tensor(0.0, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        edge_logits, state_value = actor_critic(graph, node_features)
        # Create a categorical distribution from logits.
        dist = torch.distributions.Categorical(logits=edge_logits)
        action = dist.sample()  # shape: (num_edges,)
        log_prob = dist.log_prob(action)
    
    # Enforce bidirectional consistency.
    edges_src, edges_dst = graph.edges()
    num_edges = graph.number_of_edges()
    for i in range(num_edges):
        u = edges_src[i].item()
        v = edges_dst[i].item()
        reverse_indices = ((edges_src == v) & (edges_dst == u)).nonzero(as_tuple=True)[0]
        if len(reverse_indices) > 0:
            j = reverse_indices[0].item()
            new_val = min(action[i].item(), action[j].item())
            action[i] = new_val
            action[j] = new_val

    return action, log_prob, state_value
