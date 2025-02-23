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
# import hyperparmeters  # If you need everything from hyperparmeters, consider aliasing or importing specific variables
# For instance, if hyperparmeters defines MAX_COLORS and BATCH_SIZE, you could do:
from hyperparmeters import *

# import env_help  # Alternatively, import specific helper functions:
from env_help import propagate_edge_data_bidirectional, validate_action, take_action, reconcile_graph_edge_colors, extract_node_features, reset

# import gymenv
from gymenv import *

# import create_graph  # Import specific functions if possible:
from create_graph import create_bidirectional_connected_graph, convert_to_undirected

# import actor_critic  # Import specific classes/functions from actor_critic:
from actor_critic import ActorCritic, get_action

# import compute_advantages
from compute_advantages import compute_advantages





def collect_trajectories(env, actor_critic, num_steps, device):
    """
    Run the environment using the current policy (actor_critic) and collect transitions.
    
    Parameters:
        env (EdgeColoringEnv): An instance of your Gym environment.
        actor_critic (ActorCritic): The actor-critic network.
        num_steps (int): Total number of steps (transitions) to collect.
        device (torch.device): Device on which to run computations.
    
    Returns:
        trajectories (dict): Dictionary with keys 'states', 'actions', 'rewards',
                             'next_states', 'dones', 'log_probs', and 'state_values',
                             each containing a list of corresponding transition data.
    """
    # Initialize the trajectory dictionary.
    trajectories = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': [],
        'log_probs': [],
        'state_values': []
    }
    
    # Reset the environment to get the initial observation.
    state, reward, done = env.reset()
    previous_state_reward = reward #initial reward is 0 so declaring it 
    steps_collected = 0
    while steps_collected < num_steps:
        # Get current node features from the environment's state.
        # (Assuming that env.observation is the node feature tensor.)
        node_features = env.observation  # or use env.graph.ndata['features'] if preferred
        
        # Use the actor-critic to sample an action for the current state.
        
        action, log_prob, state_value = get_action(env.graph, node_features, actor_critic, device)
        # previous reward 
        # previous_state_reward = reward
        # Take a step in the environment using the sampled action.
        next_state, reward, done, info = env.step(action)
        current_state_reward = reward
        reward_for_this_step = current_state_reward - previous_state_reward
        # print(f"previous reward : {previous_state_reward}")
        # print(f"current reward : {current_state_reward}")
        # print(f"reward_for_this_step : {reward_for_this_step}")

        previous_state_reward = current_state_reward # update new reward 

        # Store the transition.
        trajectories['states'].append(state)
        trajectories['actions'].append(action)
        trajectories['rewards'].append(reward_for_this_step)
        trajectories['next_states'].append(next_state)
        trajectories['dones'].append(done)
        trajectories['log_probs'].append(log_prob)
        trajectories['state_values'].append(state_value)
        
        # Update the current state and increment the step count.
        state = next_state
        steps_collected += 1
        
        # If episode is done, reset the environment.
        if done:
            state, reward, done = env.reset()
            previous_state_reward = reward
            #break
    
    return trajectories


# Example usage:
if __name__ == "__main__":
    # Assume the following objects are already defined:
    #   - A DGL graph 'graph_dgl' and max_colors defined.
    #   - An instance of EdgeColoringEnv named 'env'.
    #   - An ActorCritic network 'actor_critic' already instantiated and moved to 'device'.
    #   - 'device' is defined (e.g., torch.device("cpu") or torch.device("cuda")).
    # Create a sample bidirectional connected graph using your functions.
    NUM_NODES = 10
    EDGE_PROB = 0.15
    
    print(f"max_colors = {max_colors}")
    
    # Assume you have a function to create a connected bidirectional graph.
    graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
    undirected_graph = convert_to_undirected(graph_nx)
    graph_dgl = dgl.from_networkx(undirected_graph)
    
    # Instantiate the environment with the DGL graph.
    env = EdgeColoringEnv(graph_dgl, max_colors)
    
    # Reset the environment.
    obs, reward , done = env.reset()
    print("Initial observation (node features):")
    print(obs)


    actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)

    num_steps = 100  # or your desired trajectory length
    trajectories = collect_trajectories(env, actor_critic, num_steps, device)
    
    # Print out some details.
    print("Trajectory collection complete!")
    print("Number of transitions collected:", len(trajectories['states']))
    # For example, print the first transition:
    print("First transition:")
    print("State:", trajectories['states'])
    print("Action:", trajectories['actions'])
    print("Reward:", trajectories['rewards'])
    print("Done:", trajectories['dones'])


    advantages, returns = compute_advantages(trajectories, gamma=0.99, lam=0.95)
    print("Advantages:", advantages)
    print("Target Returns:", returns)














# # =============================
# # 3. Trajectory Collection Function
# # =============================
# def collect_trajectories(env, actor_critic, num_steps, device):
#     """
#     Collect trajectories by interacting with the environment using the current policy.

#     Parameters:
#         env (EdgeColoringEnv): Your Gym environment.
#         actor_critic (ActorCritic): The policy network.
#         num_steps (int): Number of steps to collect.
#         device (torch.device): Device to run computations on.

#     Returns:
#         trajectories (dict): Dictionary containing:
#             'states', 'actions', 'rewards', 'next_states', 'dones', 'log_probs', 'state_values'
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
    
#     state, reward,  done = env.reset()
    
#     steps_collected = 0
#     while steps_collected < num_steps:
#         # Get node features from the current state (observation)
#         node_features = env.observation  # assumed to be a tensor
        
#         # Sample action from the actor-critic
#         action, log_prob, state_value = get_action(env.graph, node_features, actor_critic, device)
        
#         # Take a step in the environment
#         next_state, reward, done, info = env.step(action)
        
#         # Store transition
#         trajectories['states'].append(state)
#         trajectories['actions'].append(action)
#         trajectories['rewards'].append(reward)
#         trajectories['next_states'].append(next_state)
#         trajectories['dones'].append(done)
#         trajectories['log_probs'].append(log_prob)
#         trajectories['state_values'].append(state_value)
        
#         state = next_state
#         steps_collected += 1
        
#         if done:
#             state, _,  done = env.reset()
    
#     return trajectories
