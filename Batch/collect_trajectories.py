import random
import gym
from gym import spaces
import torch
import numpy as np
import dgl
import networkx as nx
import torch.nn.functional as F

# Local Module Imports (assumed available)
from hyperparmeters import *  # e.g., max_colors, in_feats, hidden_dim, num_gat_heads, num_actions, device
from create_graph import create_bidirectional_connected_graph, convert_to_undirected
from actor_critic import ActorCritic, get_action
from gymenv import * # our environment class supporting batched graphs
from compute_advantages import compute_advantages
# compute_advantages will be defined here.

###############################################
# Trajectory Collection Function
###############################################

#  def collect_trajectories(env, actor_critic, num_steps, device):
#     """
#     Run the environment using the current policy (actor_critic) and collect transitions.
    
#     This version supports asynchronous resets: if some subenvironments in a batched environment are done,
#     they are reset individually while others continue. For batched environments, each step returns rewards
#     and done flags as tensors of shape [B] (B = batch size).
    
#     The function is fully compatible with both CPU and GPU.
    
#     Parameters:
#         env (EdgeColoringEnv): An instance of your Gym environment. For batched envs, env.step() returns
#                                done as a tensor of shape [B].
#         actor_critic (ActorCritic): The actor-critic network.
#         num_steps (int): Total number of transitions to collect.
#         device (torch.device): Device on which to run computations.
    
#     Returns:
#         trajectories (dict): Dictionary with keys:
#             'states', 'actions', 'rewards', 'next_states', 'dones', 'log_probs', 'state_values'
#         For batched environments, each element is a tensor of shape [B] per time step (stacking gives [T, B]).
#         For a single environment, they are scalars (stacking gives [T]).
#     """
#     trajectories = {
#         'states': [],
#         'actions': [],
#         'rewards': [],
#         'next_states': [],
#         'dones': [],
#         'log_probs': [],
#         'state_values': []
#     }
    
#     # Reset environment and move to device.
#     state, reward, done = env.reset()  # For batched env, reward and done are tensors of shape [B].
#     state = state.to(device)
    
#     # Initialize previous reward: if batched, as tensor; else scalar.
#     if isinstance(done, torch.Tensor):
#         prev_reward = reward if isinstance(reward, torch.Tensor) else torch.tensor([reward]*done.shape[0],
#                                                                                     device=device, dtype=torch.float32)
#     else:
#         prev_reward = reward

#     steps_collected = 0
#     while steps_collected < num_steps:
#         # Get current observation (node features)
#         node_features = env.observation.to(device)
        
#         # Sample an action using current policy.
#         action, log_prob, state_value = get_action(env.graph, node_features, actor_critic, device)
        
#         # Save previous reward (for incremental computation)
#         previous_state_reward = prev_reward
        
#         # Take a step in the environment.
#         next_state, reward, done, info = env.step(action)
#         next_state = next_state.to(device)
        
#         # Compute incremental reward: works elementwise in batched mode.
#         incremental_reward = reward - previous_state_reward
        
#         # Store transition.
#         trajectories['states'].append(state)
#         trajectories['actions'].append(action)
#         trajectories['rewards'].append(incremental_reward)
#         trajectories['next_states'].append(next_state)
#         trajectories['dones'].append(done)
#         trajectories['log_probs'].append(log_prob)
#         trajectories['state_values'].append(state_value)
        
#         # Asynchronous reset: if batched, process each subenvironment separately.
#         if isinstance(done, torch.Tensor):
#             # Unbatch the current graph.
#             subgraphs = dgl.unbatch(env.graph)
#             # Convert done tensor to list.
#             done_list = done.cpu().tolist()  # List of booleans, one per subgraph.
#             # Also update previous reward for each subgraph.
#             prev_reward_list = prev_reward.cpu().tolist() if isinstance(prev_reward, torch.Tensor) else [prev_reward]*len(done_list)
#             for i, d in enumerate(done_list):
#                 if d:  # If subgraph i is done.
#                     subgraphs[i] = reset(subgraphs[i], env.max_colors, device=env.device)
#                     subgraphs[i] = extract_node_features(subgraphs[i], device=env.device)
#                     prev_reward_list[i] = 0.0  # Reset previous reward for that subenvironment.
#             # Reassemble the subgraphs.
#             env.graph = dgl.batch(subgraphs)
#             env.observation = env.graph.ndata['features'].to(device)
#             prev_reward = torch.tensor(prev_reward_list, device=device, dtype=torch.float32)
#         else:
#             # Single-environment case.
#             if done:
#                 env.graph = reset(env.graph, env.max_colors, device=env.device)
#                 env.graph = extract_node_features(env.graph, device=env.device)
#                 env.observation = env.graph.ndata['features'].to(device)
#                 prev_reward = 0.0
        
#         # Update state and prev_reward.
#         state = next_state
#         prev_reward = reward  # Carry over reward for the next step.
#         steps_collected += 1

#     return trajectories








###############################################
# Trajectory Collection Function
###############################################

def collect_trajectories(env, actor_critic, num_steps, device):
    """
    Run the environment using the current policy and collect transitions.
    
    Supports asynchronous resets: if some subenvironments in a batched env are done,
    they are reset individually while others continue.
    
    For batched environments, at each time step the env returns rewards and dones as tensors
    of shape [B]. These are stored over T time steps so that stacking yields tensors of shape [T, B].
    
    Parameters:
        env (EdgeColoringEnv): An instance of your Gym environment. In batched mode, 
                               env.step() returns done as a tensor of shape [B].
        actor_critic (ActorCritic): The actor-critic network.
        num_steps (int): Total number of transitions to collect.
        device (torch.device): The device (CPU/GPU).
    
    Returns:
        trajectories (dict): A dictionary with keys:
            'states', 'actions', 'rewards', 'next_states', 'dones', 'log_probs', 'state_values'
        For batched env, each element (except states, which is typically complex) is a tensor of shape [B] per time step.
    """
    trajectories = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': [],
        'log_probs': [],
        'state_values': []
    }
    
    # Reset environment.
    state, reward, done = env.reset()  # In batched mode, reward and done are tensors of shape [B].
    state = state.to(device)
    
    # Initialize previous reward: tensor in batched mode, scalar in single env.
    if isinstance(done, torch.Tensor):
        prev_reward = reward if isinstance(reward, torch.Tensor) else torch.tensor([reward]*done.shape[0], device=device, dtype=torch.float32)
    else:
        prev_reward = reward

    steps_collected = 0
    while steps_collected < num_steps:
        node_features = env.observation.to(device)
        # Sample action from policy.
        action, log_prob, state_value = get_action(env.graph, node_features, actor_critic, device)
        previous_state_reward = prev_reward  # Save previous cumulative reward.
        
        # Take a step.
        next_state, reward, done, info = env.step(action)
        next_state = next_state.to(device)
        
        # Compute incremental reward (for cumulative reward signals).
        incremental_reward = reward - previous_state_reward
        
        # Store transition.
        trajectories['states'].append(state)
        trajectories['actions'].append(action)
        trajectories['rewards'].append(incremental_reward)
        trajectories['next_states'].append(next_state)
        trajectories['dones'].append(done)
        trajectories['log_probs'].append(log_prob)
        trajectories['state_values'].append(state_value)
        
        # Asynchronous reset: if batched, reset only finished subenvironments.
        if isinstance(done, torch.Tensor):
            subgraphs = dgl.unbatch(env.graph)
            done_list = done.cpu().tolist()  # List of booleans, length B.
            prev_reward_list = prev_reward.cpu().tolist() if isinstance(prev_reward, torch.Tensor) else [prev_reward]*len(done_list)
            for i, d in enumerate(done_list):
                if d:  # This subenvironment is done.
                    subgraphs[i] = reset(subgraphs[i], env.max_colors, device=env.device)
                    subgraphs[i] = extract_node_features(subgraphs[i], device=env.device)
                    prev_reward_list[i] = 0.0  # Reset its previous reward.
            env.graph = dgl.batch(subgraphs)
            env.observation = env.graph.ndata['features'].to(device)
            prev_reward = torch.tensor(prev_reward_list, device=device, dtype=torch.float32)
        else:
            # Single env.
            if done:
                env.graph = reset(env.graph, env.max_colors, device=env.device)
                env.graph = extract_node_features(env.graph, device=env.device)
                env.observation = env.graph.ndata['features'].to(device)
                prev_reward = 0.0
        
        state = next_state
        prev_reward = reward  # Carry current cumulative reward.
        steps_collected += 1

    return trajectories


###############################################
# Main Testing Script
###############################################
if __name__ == "__main__":
    # Toggle to use batched graphs or a single graph.
    use_batched = True  # Set to False to test a single graph.
    
    print(f"max_colors = {max_colors}")
    
    if use_batched:
        # Create a list of graphs for batching.
        num_graphs = 5 # Example: 3 graphs in the batch.
        graphs_list = []
        for i in range(num_graphs):
            NUM_NODES = random.randint(6, 7)
            EDGE_PROB = random.uniform(0.17, 0.2)
            graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
            undirected_graph = convert_to_undirected(graph_nx)
            graph_dgl = dgl.from_networkx(undirected_graph)
            graphs_list.append(graph_dgl)
        env = EdgeColoringEnv(graphs_list, max_colors, device=device)
    else:
        # Create a single graph.
        NUM_NODES = random.randint(6, 7)
        EDGE_PROB = random.uniform(0.17, 0.2)
        graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
        undirected_graph = convert_to_undirected(graph_nx)
        graph_dgl = dgl.from_networkx(undirected_graph)
        env = EdgeColoringEnv(graph_dgl, max_colors, device=device)
    
    # Reset the environment.
    obs, reward, done = env.reset()
    print("Initial observation (node features):")
    print(obs)
    
    # Instantiate the actor-critic network.
    actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)
    
    # Collect trajectories.
    num_steps = 100  # Desired trajectory length.
    trajectories = collect_trajectories(env, actor_critic, num_steps, device)
    
    # Print trajectory details.
    print("Trajectory collection complete!")
    print("Number of transitions collected:", len(trajectories['states']))
    print("First transition:")
    print("State:", trajectories['states'])
    print("Action:", trajectories['actions'])
    print("Reward:", trajectories['rewards'])
    print("Done:", trajectories['dones'])
    
    # Compute advantages and returns.
    advantages, returns = compute_advantages(trajectories, gamma=0.99, lam=0.95, device=device)
    print("Advantages:", advantages)
    print("Target Returns:", returns)
