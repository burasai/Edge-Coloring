import gym
from gym import spaces
import torch
import dgl
import networkx as nx
import random
import torch
import torch.nn.functional as F


def propagate_edge_data_bidirectional(graph, max_colors, device = None):
    """
    Propagate one-hot encoded edge data to nodes in both directions (src -> dst and dst -> src).
    """
    if device is None:
        device = torch.device("cpu")
    # Ensure the graph is moved to the desired device.
    graph = graph.to(device)

    num_edges = graph.number_of_edges()

    # Create one-hot encoding for edge colors
    edge_colors = graph.edata['color'].to(device)  # Colors include 0 as a valid action
    num_classes = max_colors + 1  # Include 0 in the one-hot encoding
    one_hot_vectors = torch.zeros(num_edges, num_classes, device=edge_colors.device)
    one_hot_vectors.scatter_(1, edge_colors.unsqueeze(1), 1)  # Generate one-hot vectors
    graph.edata['one_hot'] = one_hot_vectors.to(device)

    
    # Pass one-hot encoded data to both src and dst
    graph.update_all(
        dgl.function.copy_e('one_hot', 'edge_color'),  # Copy edge data to connected nodes
        dgl.function.sum('edge_color', 'received_data')  # Sum received one-hot vectors at nodes stored received_data parameter
    )

    return graph


# help function just for validating the action (bidirectionally true/same)
def validate_action(graph, action, device = None):
    """
    Validate that the action is valid by ensuring bidirectional consistency:
    If an edge (a, b) is assigned a color, its reverse edge (b, a) must have the same color.
    
    Parameters:
        graph_dgl (DGLGraph): The input graph.
        action (torch.Tensor): The action tensor to validate.
    
    Returns:
        bool: True if the action is valid, False otherwise.
    """
    if device is None:
        device = torch.device("cpu")
    # Ensure the graph is moved to the desired device.
    graph = graph.to(device)
    action = action.to(device)




    edges_src, edges_dst = graph.edges()
    valid_action = True

    for i in range(graph.number_of_edges()):
        src, dst = edges_src[i].item(), edges_dst[i].item()
        reverse_idx = ((edges_src == dst) & (edges_dst == src)).nonzero(as_tuple=True)
        if len(reverse_idx[0]) > 0:  # Reverse edge exists
            reverse_color = action[reverse_idx[0][0]].item()
            if action[i].item() != reverse_color:
                print(f"Invalid Action: Edge ({src}, {dst}) Color = {action[i].item()}, "
                      f"Reverse Edge ({dst}, {src}) Color = {reverse_color}")
                valid_action = False

    return valid_action


# special case function
def reconcile_graph_edge_colors(graph, device):
    """
    For each undirected edge in the DGL graph g (processed once),
    check if both endpoints have exactly 2 incident colors (ignoring 0).
    If their incident color sets have no intersection, then:
      - For src: let h_src = max(node_colors[src])
      - For dst: let h_dst = max(node_colors[dst])
      - Let h_high = max(h_src, h_dst) and h_low = min(h_src, h_dst).
    Then, go through all edge colors in graph.edata['color'] and replace any occurrence of h_high with h_low.
    Finally, this effectively reconciles the conflict for that edge.
    
    Assumes:
      - graph.edata['color'] is a tensor of edge colors.
      - The node incident colors can be computed from the edge colors (ignoring 0).
    
    Returns:
      Updated DGL graph graph
    """

    if device is None:
        device = torch.device("cpu")
    # Ensure the graph is moved to the desired device.
    graph = graph.to(device)



    # Compute incident colors per node (ignore color 0)
    node_incident = {node: set() for node in range(graph.num_nodes())}
    edges_src, edges_dst = graph.edges()
    num_edges = graph.number_of_edges()
    edge_colors = graph.edata['color']
    
    for i in range(num_edges):
        u = edges_src[i].item()
        v = edges_dst[i].item()
        c = edge_colors[i].item()
        if c == 0:
            continue
        node_incident[u].add(c)
        node_incident[v].add(c)
    
    # print("\n[Reconciliation] Initial Node Incident Colors:")
    # for node, colors in node_incident.items():
    #     print(f" Node {node}: {colors}")
    
    # Process each edge only once (undirected)
    processed_edges = set()
    for i in range(num_edges):
        u = edges_src[i].item()
        v = edges_dst[i].item()
        edge_key = tuple(sorted((u, v)))
        if edge_key in processed_edges:
            continue
        processed_edges.add(edge_key)
        
        colors_u = node_incident[u]
        colors_v = node_incident[v]
        # Check only if both nodes have exactly 2 incident colors
        if len(colors_u) == 2 and len(colors_v) == 2:
            inter = colors_u.intersection(colors_v)
            if inter:
                # Common color exists; no reconciliation needed.
                continue
            else:
                # No common color: need to reconcile.
                h_src = max(colors_u)
                h_dst = max(colors_v)
                if h_src >= h_dst:
                    h_high = h_src
                    h_low = h_dst
                else:
                    h_high = h_dst
                    h_low = h_src
                # print(f"\n[Reconciliation] Edge {edge_key}:")
                # print(f"  Node {u} incident colors: {colors_u}")
                # print(f"  Node {v} incident colors: {colors_v}")
                # print(f"  Replacing color {h_high} with {h_low} globally.")
                
                # Update global edge colors:
                new_edge_colors = []
                for j in range(num_edges):
                    col = edge_colors[j].item()
                    if col == h_high:
                        new_edge_colors.append(h_low)
                    else:
                        new_edge_colors.append(col)
                # Update the tensor:
                graph.edata['color'] = torch.tensor(new_edge_colors, dtype=torch.int64, device= device)
                # Also update our local node_incident dictionary:
                for node in node_incident:
                    if h_high in node_incident[node]:
                        node_incident[node].remove(h_high)
                        node_incident[node].add(h_low)
                # print(f"  Updated node incident colors:")
                # print(f"   Node {u}: {node_incident[u]}")
                # print(f"   Node {v}: {node_incident[v]}")
    
    # print("\n[Reconciliation] Final Node Incident Colors:")
    # for node, colors in node_incident.items():
    #     print(f" Node {node}: {colors}")
    return graph


def extract_node_features(graph, device = None):
    """
    Extract node features from the DGL graph to be used by the actor–critic network.
    
    This function assumes that:
      - The graph has already had its edge data propagated using one-hot encoding
        (via propagate_edge_data_bidirectional), so that each node has a 'received_data'
        field in its ndata.
      - We want to include additional structural features, such as the node degree.
    
    It constructs a feature tensor for each node by concatenating:
      1. The node's 'received_data' (aggregated one-hot encoded edge colors).
      2. The node's degree (as a scalar).
    
    The resulting features are stored in graph.ndata['features'] and the updated graph is returned.
    """
    # Get the number of nodes in the graph

    if device is None:
        device = torch.device("cpu")
    # Ensure the graph is moved to the desired device.
    graph = graph.to(device)    
    num_nodes = graph.num_nodes()
    
    # Retrieve the aggregated one-hot data from edge propagation.
    # Expected shape: (num_nodes, feature_dim) where feature_dim = num_actions (including 0).
    received_data = graph.ndata['received_data'].to(device)
    
    # Compute node degrees (for an undirected graph, in_degrees() equals out_degrees())
    # Convert degrees to float and add a new dimension so shape becomes (num_nodes, 1)
    degrees = graph.in_degrees().float().unsqueeze(1).to(received_data.device)
    
    # Concatenate received_data with the degree information to form the final feature vector.
    features = torch.cat([received_data, degrees], dim=1)
    
    # Store the features in the graph's node data for later use by the actor–critic model.
    graph.ndata['features'] = features.to(device)
    # print("Extracted node features with shape:", features.shape)
    return graph



def reset(graph, max_colors, device = None):

    if device is None:
        device = torch.device("cpu")
    # Ensure the graph is moved to the desired device.
    graph = graph.to(device)    
    graph.edata['color'] = torch.zeros(graph.number_of_edges(), dtype=torch.int64, device=device)
    graph = propagate_edge_data_bidirectional(graph, max_colors)
    graph = extract_node_features(graph)

    return graph



def take_action(graph, action, device = None):
    """
    Assigns actions to edges in the graph based on the condition where graph_dgl.edata['color'] == 1.
    
    Parameters:
        graph_dgl (DGLGraph): The input graph.
        action (torch.Tensor): A tensor representing actions to be assigned to edges.
    """
    # Create a mask where graph_dgl.edata['color'] == 0 (the uncolored edges)

    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)    
    action = action.to(device)


    mask = graph.edata['color'] == 0

    # Assign actions to edges where the mask is True (assigning the colors to the uncolored edges only)
    graph.edata['color'][mask] = action[mask]

    # Debug: Print updated edge colors
    # edges_src, edges_dst = graph.edges()
    # edge_colors = graph.edata['color']
    # print("\nUpdated Edge Colors After Action:")
    # for i in range(graph.number_of_edges()):
    #     print(f"Edge ({edges_src[i].item()}, {edges_dst[i].item()}): Color {edge_colors[i].item()}")
    return graph




def find_uncolorable_edges(graph, max_colors, device=None):
    """
    For each uncolored edge (where graph.edata['color'] == 0), this function:
      - Computes each node's incident color set (from colored edges only, ignoring color 0),
        represented as a binary (multi-hot) vector of length (max_colors+1).
      - Considers only those uncolored edges whose endpoints each have exactly 2 incident colors 
        (ignoring index 0).
      - Checks if the incident color sets of the two endpoints have no intersection.
    If both conditions hold for an edge, it is marked as violating.
    
    Returns:
      violating_edge_indices: a tensor containing the indices of violating edges.
      incident_colors_src: a tensor containing the binary incident color vector for the source nodes
                           of the violating edges.
      incident_colors_dst: a tensor containing the binary incident color vector for the destination nodes
                           of the violating edges.
    """
    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)
    
    num_nodes = graph.num_nodes()
    num_edges = graph.number_of_edges()
    
    # --- Step 1: Build a node incident color matrix ---
    # Use the edge color feature to create one-hot vectors for each edge.
    # Note: For edges with color == 0 (uncolored), we ignore their contribution.
    edge_colors = graph.edata['color'].to(device)  # shape: (num_edges,)
    one_hot = F.one_hot(edge_colors, num_classes=max_colors+1).float()  # shape: (num_edges, max_colors+1)
    # Create a mask to zero out contributions from uncolored edges.
    mask = (edge_colors != 0).unsqueeze(1).float()  # shape: (num_edges, 1)
    one_hot = one_hot * mask
    
    # Save the one-hot encoded colored-edge info in the graph.
    graph.edata['one_hot'] = one_hot
    
    # Propagate these edge one-hot vectors to nodes.
    # This sums the contributions from all incident edges.
    graph.update_all(
        dgl.function.copy_e('one_hot', 'm'),
        dgl.function.sum('m', 'node_colors')
    )
    
    # Convert to a binary indicator: if a node has at least one incident edge with that color, mark as 1.
    # node_incident: (num_nodes, max_colors+1)
    node_incident = (graph.ndata['node_colors'] > 0).float()
    
    # --- Step 2: Filter for candidate (uncolored) edges ---
    # We only consider edges that are uncolored.
    candidate_mask = (edge_colors == 0)
    candidate_edges = torch.nonzero(candidate_mask, as_tuple=False).squeeze()  # indices of candidate edges
    
    # Get the endpoints for every edge in the graph.
    src, dst = graph.edges()
    # Filter the endpoints corresponding to our candidate edges.
    candidate_src = src[candidate_edges]
    candidate_dst = dst[candidate_edges]
    
    # --- Step 3: Extract and process incident colors for candidate edge endpoints ---
    # We ignore the 0-th column (assumed to be "uncolored") in further checks.
    src_incident = node_incident[candidate_src][:, 1:]  # shape: (num_candidate_edges, max_colors)
    dst_incident = node_incident[candidate_dst][:, 1:]
    
    # Condition 1: Each endpoint must have exactly 2 incident colors.
    cond1 = (src_incident.sum(dim=1) == 2) & (dst_incident.sum(dim=1) == 2)
    
    # Condition 2: The intersection of the incident color sets is empty.
    # (Elementwise multiplication gives 1 only for common colors.)
    intersection = (src_incident * dst_incident).sum(dim=1)
    cond2 = (intersection == 0)
    
    # Combine conditions to get a mask for violating candidate edges.
    violating_mask = cond1 & cond2
    violating_edge_indices = candidate_edges[violating_mask]
    
    # Also extract the incident color vectors for the endpoints of these violating edges.
    violating_src = candidate_src[violating_mask]
    violating_dst = candidate_dst[violating_mask]
    
    incident_colors_src = node_incident[violating_src]
    incident_colors_dst = node_incident[violating_dst]
    
    return violating_edge_indices, incident_colors_src, incident_colors_dst
