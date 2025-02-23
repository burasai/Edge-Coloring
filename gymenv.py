# Standard Library Imports
import random

# Third-Party Imports
import gym
from gym import spaces
import torch
import numpy as np
import dgl
import networkx as nx

# Local Module Imports
import hyperparmeters  # If you need everything from hyperparmeters, consider aliasing or importing specific variables
# For instance, if hyperparmeters defines MAX_COLORS and BATCH_SIZE, you could do:
from hyperparmeters import *

import env_help  # Alternatively, import specific helper functions:
from env_help import propagate_edge_data_bidirectional, validate_action, take_action, reconcile_graph_edge_colors, extract_node_features, reset

import create_graph  # Import specific functions if possible:
from create_graph import create_bidirectional_connected_graph, convert_to_undirected

import actor_critic  # Import specific classes/functions from actor_critic:
from actor_critic import ActorCritic, get_action



# Import your helper functions from your existing code.
# (Make sure these functions are defined in the same module or are importable.)
# from your_module import propagate_edge_data_bidirectional, take_action, reconcile_graph_edge_colors, extract_node_features, reset

# For the sake of this example, we assume the functions are defined above.

class EdgeColoringEnv(gym.Env):
    """
    Gym environment for the edge-coloring problem:
    - Each edge must be assigned a color.
    - Each node can have at most two unique incident colors.
    - The objective is to color all edges while maximizing the number of unique colors used.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, graph_dgl, max_colors):
        """
        Initialize the environment.
        
        Parameters:
            graph_dgl (DGLGraph): A pre-created DGL graph.
            max_colors (int): Maximum number of colors available (excluding 0 for uncolored).
        """
        super(EdgeColoringEnv, self).__init__()
        
        self.graph = graph_dgl
        self.max_colors = max_colors
        self.num_edges = self.graph.number_of_edges()

        # Action space: For each edge, an integer in [0, max_colors]
        # 0 indicates uncolored; available colors are 1..max_colors.
        self.action_space = spaces.MultiDiscrete([max_colors + 1] * self.num_edges)
        
        # Reset the graph to the initial state.
        self.graph = reset(self.graph, self.max_colors)
        self.graph = extract_node_features(self.graph)
        self.observation = self.graph.ndata['features']
        
        
        # Define observation space.
        # Here we use a Box space with shape equal to the node feature tensor shape.
        # (You may need to adjust low/high values according to your feature range.)
        obs_shape = self.observation.shape  # e.g., (num_nodes, feature_dim)
        # print(f"obs: {self.observation}  \n obs_shape: {obs_shape}")
        # self.observation_space = spaces.Box(low=0, high=float(max_colors), shape=obs_shape, dtype=torch.float32)
        self.observation_space = spaces.Box(low=0, high=float(max_colors), shape=obs_shape, dtype=np.float32)


    def reset(self):
        """
        Reset the environment to an initial uncolored graph state.
        
        Returns:
            observation (Tensor): Node features extracted from the reset graph.
        """
        # print(f"in reset function ")
        self.graph = reset(self.graph, self.max_colors)
        # print(f" after reset the graph : {self.graph.edata['color']} ")
        self.graph = extract_node_features(self.graph)
        self.observation = self.graph.ndata['features']
        # print(f"reset graph features {self.observation}")
        done = not (self.graph.edata['color'] == 0).any().item()

        unique_colors = self.graph.edata['color'].unique().tolist()
        if 0 in unique_colors:
            unique_colors.remove(0)
        reward = len(unique_colors)
        # print(f"Reward: {reward} (Unique Colors: {unique_colors})")
        return self.observation, reward,  done

    def step(self, action):
        """
        Apply the given action to the graph, update its state, compute the reward, and return the new state.
        
        Parameters:
            action (Tensor or np.array): A tensor representing the action (color assignment for each edge).
            
        Returns:
            observation (Tensor): The updated node features.
            reward (float): The reward computed (number of unique colors used).
            done (bool): True if the episode is finished (all edges are colored), else False.
            info (dict): Additional diagnostic information.
        """
        # Convert action to a torch tensor if necessary.
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.int64)
        
        # Backup current edge colors.
        old_colors = self.graph.edata['color'].clone()
        
        # Apply action: update edge colors where uncolored.
        self.graph = take_action(self.graph, action)
        # Propagate edge data to nodes.
        self.graph = propagate_edge_data_bidirectional(self.graph, self.max_colors)
        
        # Validate state: check incident color counts.
        new_node_info = self.graph.ndata['received_data']
        color_matrix = new_node_info[:, 1:]  # exclude column 0 (uncolored)
        color_counts = (color_matrix > 0).sum(dim=1)
        # print("Color counts per node:", color_counts.tolist())
        
        # Identify nodes violating the constraint (>2 colors)
        violating_nodes = (color_counts > 2).nonzero(as_tuple=True)[0]
        # print(f"violating nodes {violating_nodes}")
        if len(violating_nodes) > 0:
            # print("Violating nodes:", violating_nodes.tolist())
            # Revert edge colors for these nodes.
            edges_src, edges_dst = self.graph.edges()
            for node in violating_nodes:
                connected_edges = ((edges_src == node) | (edges_dst == node)).nonzero(as_tuple=True)[0]
                for edge_idx in connected_edges:
                    self.graph.edata['color'][edge_idx] = old_colors[edge_idx]
            # Re-propagate updated edge data.
            self.graph = propagate_edge_data_bidirectional(self.graph, self.max_colors)
        
        # Handle special conflict cases by reconciliation.
        self.graph = reconcile_graph_edge_colors(self.graph)
        # Re-propagate again.
        self.graph = propagate_edge_data_bidirectional(self.graph, self.max_colors)
        
        # Check if all edges are colored (nonzero).
        done = not (self.graph.edata['color'] == 0).any().item()
        if done:
            print("All edges are colored.")
        else:
            print("Some edges remain uncolored.")
        
        # Compute reward: number of unique colors used (excluding 0).
        unique_colors = self.graph.edata['color'].unique().tolist()
        if 0 in unique_colors:
            unique_colors.remove(0)
        reward = len(unique_colors)
        # print(f"Reward: {reward} (Unique Colors: {unique_colors})")
        
        # Update observation.
        self.graph = extract_node_features(self.graph)
        self.observation = self.graph.ndata['features']
        info = {}  # optional info can be added here.
        
        return self.observation, reward, done, info

    def render(self, mode='human'):
        """
        Render the current state of the environment.
        For simplicity, prints the edge colors and node features.
        """
        print("Edge Colors:", self.graph.edata['color'])
        print("Node Features:", self.graph.ndata['features'])

    def close(self):
        pass

    




# Example usage of the environment:
if __name__ == "__main__":
    # Create a sample bidirectional connected graph using your functions.
    NUM_NODES = 6
    EDGE_PROB = 0.2
    
    print(f"max_colors = {max_colors}")
    
    # Assume you have a function to create a connected bidirectional graph.
    graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
    undirected_graph = convert_to_undirected(graph_nx)
    graph_dgl = dgl.from_networkx(undirected_graph)
    
    # Instantiate the environment with the DGL graph.
    env = EdgeColoringEnv(graph_dgl, max_colors)
    
    # Reset the environment.
    obs, reward, done= env.reset()
    print("Initial observation (node features):")
    print(obs)


    actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)

    graph_dgl = extract_node_features(graph_dgl)
    node_features = graph_dgl.ndata['features']
    action, log_prob, state_value = get_action(graph_dgl, node_features, actor_critic, device)
    
    step_count = 0
    while not done:
        step_count += 1
        print(f"\n=== Step {step_count} ===")
        if step_count > 100 :
            break
        # Generate a random valid action
        # _, random_action = generate_valid_action(graph_dgl, MAX_COLORS)
        action, log_prob, state_value = get_action(graph_dgl, node_features, actor_critic, device)
        print("Sampled action for each edge:", action.tolist())
        print("Log probabilities:", log_prob.tolist())
        print("State value:", state_value.item())

        print(f"\n Action Tensor for Step {step_count}:", action.tolist())

        # Validate the random action
        if validate_action(graph_dgl, action):
            print(f"\nThe action for Step {step_count} is valid!")
        else:
            print(f"\nThe action for Step {step_count} is invalid!")

        # Take a step with the random action
        print("\nTaking the step...")
        print("Before Action \n graph_dgl.edata['color']:", graph_dgl.edata['color'])
        # graph_dgl, done, reward = step(graph_dgl, action, MAX_COLORS)
        observation, reward, done, info = env.step(action)
        print("===========================================================================================")
        print(f"observation:{observation} \n\n reward: {reward} \n\n done: {done} \n\n info: {info}")
        print("===========================================================================================")


        # getting node features
        graph_dgl = extract_node_features(graph_dgl)

        print("After Action \n graph_dgl.edata['color']:", graph_dgl.edata['color'])

        if done:
            print("\nAll edges are successfully colored!")
        else:
            print("\nSome edges are still not colored. Continuing to the next step...")


        
        
    
    # env.step(action)
        # print(f"graph {graph_dgl.edata['color']}")
        # env.reset()

    
    # Sample a random action from the action space.
    # random_action = env.action_space.sample()
    # print("Random action:", random_action)
    
    # # Take a step in the environment with the random action.
    # next_obs, reward, done, info = env.step(random_action)
    # print("Next observation (node features):")
    # print(next_obs)
    # print("Reward:", reward)
    # print("Done:", done)
    # print("Info:", info)
    
    # # Optionally render the environment.
    # env.render()


