# Standard Library Imports
import random

# Third-Party Imports
import gym
from gym import spaces
import torch
import numpy as np
import dgl
import networkx as nx
import torch.nn.functional as F


# Local Module Imports
# import hyperparmeters  # If you need everything from hyperparmeters, consider aliasing or importing specific variables
# # For instance, if hyperparmeters defines MAX_COLORS and BATCH_SIZE, you could do:
from hyperparmeters import *

# import env_help  # Alternatively, import specific helper functions:
# from env_help import propagate_edge_data_bidirectional, validate_action, take_action, reconcile_graph_edge_colors, extract_node_features, reset

# import create_graph  # Import specific functions if possible:
from create_graph import create_bidirectional_connected_graph, convert_to_undirected

# import actor_critic  # Import specific classes/functions from actor_critic:
from actor_critic import ActorCritic, get_action



# Import your helper functions from your existing code.
# (Make sure these functions are defined in the same module or are importable.)
# from your_module import propagate_edge_data_bidirectional, take_action, reconcile_graph_edge_colors, extract_node_features, reset

# For the sake of this example, we assume the functions are defined above.


# # Local Module Imports
# import hyperparmeters
# from hyperparmeters import *
# import env_help
# from env_help import propagate_edge_data_bidirectional, validate_action, take_action, reconcile_graph_edge_colors, extract_node_features, reset
# import create_graph
# from create_graph import create_bidirectional_connected_graph, convert_to_undirected
# import actor_critic
# from actor_critic import ActorCritic, get_action

#####################################
# Modified Helper Functions
#####################################
def propagate_edge_data_bidirectional(graph, max_colors, device=None):
    """
    Propagate one-hot encoded edge color data to nodes in both directions (src->dst and dst->src).

    Args:
        graph (DGLGraph): Input graph.
        max_colors (int): Maximum number of colors available.
        device (torch.device, optional): Device for computation. Defaults to CPU.
        
    Returns:
        DGLGraph: Graph with updated node 'received_data' (sum of one-hot encoded edge colors).
    """
    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)
    
    # Number of edges and one-hot encoding dimensions.
    num_edges = graph.number_of_edges()
    edge_colors = graph.edata['color'].to(device)
    num_classes = max_colors + 1  # Include color 0.
    
    # Create one-hot vectors for each edge's color.
    one_hot_vectors = torch.zeros(num_edges, num_classes, device=edge_colors.device)
    one_hot_vectors.scatter_(1, edge_colors.unsqueeze(1), 1)
    graph.edata['one_hot'] = one_hot_vectors.to(device)

    # Propagate one-hot vectors from edges to nodes using DGL's update_all.
    graph.update_all(
        dgl.function.copy_e('one_hot', 'edge_color'),  # Copy one-hot vectors.
        dgl.function.sum('edge_color', 'received_data')   # Sum vectors at nodes.
    )
    return graph

def validate_action(graph, action, device=None):
    """
    Validate that the provided action is consistent:
      - For each edge, the action (color) should match that of its reverse edge.

    This function supports both single and batched graphs. For batched graphs, it unbatches
    and validates each subgraph separately.

    Args:
        graph (DGLGraph): Input graph (or batched graph).
        action (Tensor): Action tensor for edges.
        device (torch.device, optional): Device to use.
    
    Returns:
        bool: True if actions are valid; False otherwise.
    """
    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)
    action = action.to(device)
    
    # Check if the graph is batched using a custom attribute "batch_size" (default=1)
    batch_size = getattr(graph, 'batch_size', 1)
    if batch_size > 1:
        # Unbatch and split action tensor accordingly.
        subgraphs = dgl.unbatch(graph)
        valid = True
        edges_per_subgraph = [g.number_of_edges() for g in subgraphs]
        action_split = torch.split(action, edges_per_subgraph)
        for g, a in zip(subgraphs, action_split):
            valid = valid and validate_action(g, a, device)
        return valid

    # Single-graph branch: Iterate through each edge and check reverse edge.
    edges_src, edges_dst = graph.edges()
    valid_action_flag = True
    for i in range(graph.number_of_edges()):
        src, dst = edges_src[i].item(), edges_dst[i].item()
        # Find reverse edge indices.
        reverse_idx = ((edges_src == dst) & (edges_dst == src)).nonzero(as_tuple=True)
        if len(reverse_idx[0]) > 0:
            reverse_color = action[reverse_idx[0][0]].item()
            if action[i].item() != reverse_color:
                print(f"Invalid Action: Edge ({src}, {dst}) Color = {action[i].item()}, "
                      f"Reverse Edge ({dst}, {src}) Color = {reverse_color}")
                valid_action_flag = False
    return valid_action_flag

def reconcile_graph_edge_colors(graph, device):
    """
    Reconcile edge colors in the graph if there is a conflict:
      For each undirected edge where each endpoint has exactly 2 colors (but no intersection),
      replace the higher color with the lower one.
    
    This function supports batched graphs by unbatching, processing each, and re-batching.

    Args:
        graph (DGLGraph): Input graph.
        device (torch.device): Device for computation.
    
    Returns:
        DGLGraph: Graph with reconciled edge colors.
    """
    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)
    
    # If batched, unbatch, process each, then batch again.
    subgraphs = dgl.unbatch(graph)
    if len(subgraphs) > 1:
        reconciled = [reconcile_graph_edge_colors(g, device) for g in subgraphs]
        return dgl.batch(reconciled)
    
    # Single graph reconciliation.
    node_incident = {node: set() for node in range(graph.num_nodes())}
    edges_src, edges_dst = graph.edges()
    num_edges = graph.number_of_edges()
    edge_colors = graph.edata['color']
    
    # Build incident color sets for each node.
    for i in range(num_edges):
        u = edges_src[i].item()
        v = edges_dst[i].item()
        c = edge_colors[i].item()
        if c == 0:
            continue
        node_incident[u].add(c)
        node_incident[v].add(c)
    
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
        if len(colors_u) == 2 and len(colors_v) == 2:
            inter = colors_u.intersection(colors_v)
            if inter:
                continue
            else:
                h_src = max(colors_u)
                h_dst = max(colors_v)
                if h_src >= h_dst:
                    h_high, h_low = h_src, h_dst
                else:
                    h_high, h_low = h_dst, h_src
                new_edge_colors = []
                for j in range(num_edges):
                    col = edge_colors[j].item()
                    if col == h_high:
                        new_edge_colors.append(h_low)
                    else:
                        new_edge_colors.append(col)
                graph.edata['color'] = torch.tensor(new_edge_colors, dtype=torch.int64, device=device)
                for node in node_incident:
                    if h_high in node_incident[node]:
                        node_incident[node].remove(h_high)
                        node_incident[node].add(h_low)
    return graph

def extract_node_features(graph, device=None):
    """
    Extract node features for the actor–critic network by concatenating:
      1. Aggregated one-hot encoded edge data ('received_data').
      2. Node degree.

    Args:
        graph (DGLGraph): Input graph.
        device (torch.device, optional): Device for computation.
        
    Returns:
        DGLGraph: Graph with updated node features stored in 'features'.
    """
    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)
    num_nodes = graph.num_nodes()
    received_data = graph.ndata['received_data'].to(device)
    degrees = graph.in_degrees().float().unsqueeze(1).to(received_data.device)
    features = torch.cat([received_data, degrees], dim=1)
    graph.ndata['features'] = features.to(device)
    return graph

def reset(graph, max_colors, device=None):
    """
    Reset the graph to its initial state:
      - Set all edge colors to 0.
      - Propagate edge data and extract node features.
    
    Args:
        graph (DGLGraph): Input graph (or batched graph).
        max_colors (int): Maximum number of colors.
        device (torch.device, optional): Device for computation.
    
    Returns:
        DGLGraph: Reset graph.
    """
    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)
    graph.edata['color'] = torch.zeros(graph.number_of_edges(), dtype=torch.int64, device=device)
    graph = propagate_edge_data_bidirectional(graph, max_colors, device=device)
    graph = extract_node_features(graph, device=device)
    return graph

def take_action(graph, action, device=None):
    """
    Apply the given action (color assignment) to the graph.
    Only assigns colors to edges that are uncolored (color==0).
    
    Args:
        graph (DGLGraph): Input graph.
        action (Tensor): Action tensor (colors for edges).
        device (torch.device, optional): Device for computation.
    
    Returns:
        DGLGraph: Graph with updated edge colors.
    """
    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)
    action = action.to(device)
    mask = graph.edata['color'] == 0
    graph.edata['color'][mask] = action[mask]
    return graph

def find_uncolorable_edges(graph, max_colors, device=None):
    """
    Identify uncolored edges where both endpoints have exactly two incident colors,
    and the intersection of incident colors is empty.
    
    For batched graphs, processes each subgraph independently.
    
    Args:
        graph (DGLGraph): Input graph.
        max_colors (int): Maximum number of colors.
        device (torch.device, optional): Device for computation.
    
    Returns:
        Tuple: (violating_edge_indices, incident_colors_src, incident_colors_dst)
    """
    if device is None:
        device = torch.device("cpu")
    graph = graph.to(device)
    
    # If batched, unbatch and process each subgraph.
    subgraphs = dgl.unbatch(graph)
    if len(subgraphs) > 1:
        results = [find_uncolorable_edges(g, max_colors, device) for g in subgraphs]
        violating_edges = torch.cat([res[0] for res in results])
        incident_src = torch.cat([res[1] for res in results])
        incident_dst = torch.cat([res[2] for res in results])
        return violating_edges, incident_src, incident_dst

    num_nodes = graph.num_nodes()
    num_edges = graph.number_of_edges()
    edge_colors = graph.edata['color'].to(device)
    one_hot = F.one_hot(edge_colors, num_classes=max_colors+1).float()
    mask = (edge_colors != 0).unsqueeze(1).float()
    one_hot = one_hot * mask
    graph.edata['one_hot'] = one_hot
    graph.update_all(
        dgl.function.copy_e('one_hot', 'm'),
        dgl.function.sum('m', 'node_colors')
    )
    node_incident = (graph.ndata['node_colors'] > 0).float()
    candidate_mask = (edge_colors == 0)
    candidate_edges = torch.nonzero(candidate_mask, as_tuple=False).squeeze()
    src, dst = graph.edges()
    candidate_src = src[candidate_edges]
    candidate_dst = dst[candidate_edges]
    src_incident = node_incident[candidate_src][:, 1:]
    dst_incident = node_incident[candidate_dst][:, 1:]
    cond1 = (src_incident.sum(dim=1) == 2) & (dst_incident.sum(dim=1) == 2)
    intersection = (src_incident * dst_incident).sum(dim=1)
    cond2 = (intersection == 0)
    violating_mask = cond1 & cond2
    violating_edge_indices = candidate_edges[violating_mask]
    violating_src = candidate_src[violating_mask]
    violating_dst = candidate_dst[violating_mask]
    incident_colors_src = node_incident[violating_src]
    incident_colors_dst = node_incident[violating_dst]
    return violating_edge_indices, incident_colors_src, incident_colors_dst

###############################################
# Gym Environment Class for Edge Coloring
###############################################
class EdgeColoringEnv(gym.Env):
    """
    Gym environment for the edge-coloring problem.
    Supports both single graphs and batched graphs.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, graph_dgl, max_colors, device=torch.device("cpu")):
        """
        Initialize the environment.

        Args:
            graph_dgl (DGLGraph or list): A DGL graph or list of DGL graphs.
            max_colors (int): Maximum number of colors available.
            device (torch.device): Device for computation.
        """
        super(EdgeColoringEnv, self).__init__()
        self.device = device
        self.max_colors = max_colors

        # If graph_dgl is a list, ensure each graph is processed with reset() and extract_node_features()
        if isinstance(graph_dgl, list):
            processed_graphs = []
            for g in graph_dgl:
                g = g.to(self.device)
                g = reset(g, self.max_colors, device=self.device)
                g = extract_node_features(g, device=self.device)
                processed_graphs.append(g)
            self.graph = dgl.batch(processed_graphs).to(self.device)
        else:
            self.graph = graph_dgl.to(self.device)
            self.graph = reset(self.graph, self.max_colors, device=self.device)
            self.graph = extract_node_features(self.graph, device=self.device)

        self.num_edges = self.graph.number_of_edges()
        # Action space: each edge is assigned an integer in [0, max_colors].
        self.action_space = spaces.MultiDiscrete([max_colors + 1] * self.num_edges)
        self.observation = self.graph.ndata['features'].to(self.device)
        obs_shape = self.observation.shape
        self.observation_space = spaces.Box(low=0, high=float(max_colors), shape=obs_shape, dtype=np.float32)

    def reset(self):
        """
        Reset the environment.
        
        Returns:
            tuple: (observation, reward, done)
                - observation (Tensor): Node features.
                - reward (float): Number of unique colors used.
                - done (bool): True if all edges are colored.
        """
        self.graph = reset(self.graph, self.max_colors, device=self.device)
        self.graph = extract_node_features(self.graph, device=self.device)
        self.observation = self.graph.ndata['features'].to(self.device)
        # Define done: True only when all edges in all subgraphs are colored.
        done = not (self.graph.edata['color'] == 0).any().item()
        unique_colors = self.graph.edata['color'].unique().tolist()
        if 0 in unique_colors:
            unique_colors.remove(0)
        reward = len(unique_colors)
        print(f"Reward: {reward} (Unique Colors: {unique_colors})")
        return self.observation, reward, done

    def step(self, action):
        """
        Apply the given action to the graph, update the state, and compute reward.
        
        Args:
            action (Tensor): Action tensor (edge color assignments).
        
        Returns:
            tuple: (observation, reward, done, info)
                - observation (Tensor): Updated node features.
                - reward (float): Number of unique colors used.
                - done (bool): True if all edges are colored.
                - info (dict): Additional info (empty here).
        """
        action = action.to(self.device)
        old_colors = self.graph.edata['color'].clone().to(self.device)
        
        # Apply the action to uncolored edges.
        self.graph = take_action(self.graph, action, device=self.device)
        # Propagate new edge data.
        self.graph = propagate_edge_data_bidirectional(self.graph, self.max_colors, device=self.device)
        
        # Compute the aggregated one-hot data at nodes.
        new_node_info = self.graph.ndata['received_data'].to(self.device)
        # Exclude the 0 (uncolored) column.
        color_matrix = new_node_info[:, 1:]
        # Count the number of colors per node.
        color_counts = (color_matrix > 0).sum(dim=1)
        print("Color counts per node:", color_counts.tolist())
        
        # Identify nodes with more than 2 incident colors.
        violating_nodes = (color_counts > 2).nonzero(as_tuple=True)[0]
        print(f"Violating nodes: {violating_nodes.tolist() if len(violating_nodes) > 0 else 'None'}")
        if len(violating_nodes) > 0:
            edges_src, edges_dst = self.graph.edges()
            for node in violating_nodes:
                connected_edges = ((edges_src == node) | (edges_dst == node)).nonzero(as_tuple=True)[0]
                for edge_idx in connected_edges:
                    self.graph.edata['color'][edge_idx] = old_colors[edge_idx]
            self.graph = propagate_edge_data_bidirectional(self.graph, self.max_colors, device=self.device)
        
        # Reconcile any conflicting edge colors.
        self.graph = reconcile_graph_edge_colors(self.graph, device=self.device)
        self.graph = propagate_edge_data_bidirectional(self.graph, self.max_colors, device=self.device)
        
        done = not (self.graph.edata['color'] == 0).any().item()
        if done:
            print("All edges are colored.")
        else:
            print("Some edges remain uncolored.")
        unique_colors = self.graph.edata['color'].unique().tolist()
        if 0 in unique_colors:
            unique_colors.remove(0)
        reward = len(unique_colors)
        print(f"Reward: {reward} (Unique Colors: {unique_colors})")
        
        # Update observation.
        self.graph = extract_node_features(self.graph, device=self.device)
        self.observation = self.graph.ndata['features'].to(self.device)
        info = {}
        return self.observation, reward, done, info

    def render(self, mode='human'):
        """
        Render the current state by printing edge colors and node features.
        """
        print("Edge Colors:", self.graph.edata['color'])
        print("Node Features:", self.graph.ndata['features'])

    def close(self):
        pass

###############################################
# End of Module
###############################################


        
        
    

# Assuming all necessary functions and classes have been imported:
# create_bidirectional_connected_graph, convert_to_undirected, EdgeColoringEnv,
# ActorCritic, extract_node_features, get_action, validate_action

if __name__ == "__main__":
    # Hyperparameters for testing
    NUM_NODES = random.randint(6,7)
    EDGE_PROB = random.uniform(0.17, 0.2)
    
    num_graphs = 5  # Number of graphs to batch
    print(f"max_colors = {max_colors}")
    
    # Create a list of DGL graphs
    graphs_list = []
    for i in range(num_graphs):
        graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
        undirected_graph = convert_to_undirected(graph_nx)
        graph_dgl = dgl.from_networkx(undirected_graph)
        graphs_list.append(graph_dgl)
    
    # Instantiate the batched environment with a list of graphs.
    # The environment will use dgl.batch internally if given a list.
    env = EdgeColoringEnv(graphs_list, max_colors)
    
    # Reset the batched environment.
    obs, reward, done = env.reset()
    print("Initial batched observation (node features):")
    print(obs)
    
    # For testing, we need to define these hyperparameters for ActorCritic:
    # (Assuming they are defined in hyperparmeters or set here)
    in_feats = obs.shape[1]  # Node feature dimension
    hidden_dim = 32
    num_gat_heads = 4
    num_actions = max_colors + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the actor-critic network and move to the device.
    actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)
    
    # Ensure the batched graph has extracted node features.
    batched_graph = extract_node_features(env.graph)
    node_features = batched_graph.ndata['features']
    
    # Test get_action on the batched graph.
    action, log_prob, state_value = get_action(batched_graph, node_features, actor_critic, device)
    print("Sampled actions for each edge in the batched graph:", action.tolist())
    print("Log probabilities:", log_prob.tolist())
    print("State value:", state_value.item())
    
    step_count = 0
    while not done:
        step_count += 1
        print(f"\n=== Step {step_count} ===")
        if step_count > 100:
            break
        
        # Get new actions from the batched environment.
        action, log_prob, state_value = get_action(env.graph, env.observation, actor_critic, device)
        print("Sampled action for each edge:", action.tolist())
        print("Log probabilities:", log_prob.tolist())
        print("State value:", state_value.item())

        # Validate the action on the batched graph.
        if validate_action(env.graph, action):
            print(f"\nThe action for Step {step_count} is valid!")
        else:
            print(f"\nThe action for Step {step_count} is invalid!")

        # Take a step in the batched environment.
        print("\nTaking the step...")
        print("Before Action \n env.graph.edata['color']:", env.graph.edata['color'])
        observation, reward, done, info = env.step(action)
        print("===========================================================================================")
        print(f"Observation: {observation}\nReward: {reward}\nDone: {done}\nInfo: {info}")
        print("===========================================================================================")

        # Extract node features for the updated batched graph.
        env.graph = extract_node_features(env.graph)
        print("After Action \n env.graph.edata['color']:", env.graph.edata['color'])

        if done:
            print("\nAll edges in all graphs are successfully colored!")
        else:
            print("\nSome edges are still not colored. Continuing to the next step...")



