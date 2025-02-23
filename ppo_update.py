import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
from torch.distributions import Categorical



# Local Module Imports
import hyperparmeters  # If you need everything from hyperparmeters, consider aliasing or importing specific variables
# For instance, if hyperparmeters defines MAX_COLORS and BATCH_SIZE, you could do:
from hyperparmeters import *

# import env_help  # Alternatively, import specific helper functions:
# from env_help import propagate_edge_data_bidirectional, validate_action, take_action, reconcile_graph_edge_colors, extract_node_features, reset

# import gymenv
# from gymenv import *

# import create_graph  # Import specific functions if possible:
# from create_graph import create_bidirectional_connected_graph, convert_to_undirected

# import actor_critic  # Import specific classes/functions from actor_critic:
# from actor_critic import ActorCritic, get_action

import compute_advantages
from compute_advantages import compute_advantages




def ppo_update(actor_critic, optimizer, trajectories, clip_epsilon, epochs, batch_size):
    """
    Update the actor-critic network using PPO.
    
    Parameters:
        actor_critic: The actor-critic network.
        optimizer: The optimizer (e.g., Adam).
        trajectories: A dictionary with keys 'states', 'actions', 'rewards',
                      'next_states', 'dones', 'log_probs', 'state_values'.
        clip_epsilon: PPO clip parameter.
        epochs: Number of training epochs per update.
        batch_size: Size of mini-batches.
    """
    # Compute advantages and target returns (using GAE)
    advantages, returns = compute_advantages(trajectories)
    
    # Convert trajectory lists to tensors
    states = torch.stack(trajectories['states'])
    actions = torch.stack(trajectories['actions'])
    old_log_probs = torch.stack(trajectories['log_probs'])
    advantages = advantages.detach()
    returns = returns.detach()
    
    num_samples = states.shape[0]
    
    for epoch in range(epochs):
        # Shuffle indices for mini-batch training
        indices = torch.randperm(num_samples)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            mini_indices = indices[start:end]
            
            mini_states = states[mini_indices]
            mini_actions = actions[mini_indices]
            mini_old_log_probs = old_log_probs[mini_indices]
            mini_advantages = advantages[mini_indices]
            mini_returns = returns[mini_indices]
            
            # Get new action logits and state values from the network
            edge_logits, state_values = actor_critic(mini_states)  # Modify accordingly for your input shapes
            
            # Create a distribution and compute new log probabilities
            dist = Categorical(logits=edge_logits)
            new_log_probs = dist.log_prob(mini_actions)
            
            # Compute probability ratios
            ratio = torch.exp(new_log_probs - mini_old_log_probs)
            
            # Compute clipped surrogate loss for the actor
            actor_loss = -torch.min(ratio * mini_advantages,
                                    torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mini_advantages).mean()
            
            # Compute critic loss (mean squared error)
            critic_loss = F.mse_loss(state_values, mini_returns)
            
            # Optionally, add an entropy bonus
            entropy_loss = -dist.entropy().mean()
            
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
    return total_loss.item()















# =============================
# 2. PPO Update Function
# =============================
# def ppo_update(actor_critic, env, optimizer, trajectories, clip_epsilon, epochs, batch_size, device):
#     """
#     Perform PPO update on the actor-critic network using collected trajectories.

#     Parameters:
#         actor_critic (nn.Module): The actor-critic network.
#         optimizer (torch.optim.Optimizer): Optimizer for updating network parameters.
#         trajectories (dict): Collected transitions containing keys:
#             'states', 'actions', 'log_probs', 'state_values', 'rewards', 'dones'
#         clip_epsilon (float): Clipping parameter for PPO.
#         epochs (int): Number of training epochs over the trajectory.
#         batch_size (int): Mini-batch size.
#         device (torch.device): Device to run computation on.

#     Returns:
#         total_loss (float): The final loss value for logging.
#     """
#     advantages, returns = compute_advantages(trajectories)
    
#     # Convert lists to tensors (if not already)
#     states = torch.stack(trajectories['states']).to(device)          # shape: (N, ...)
#     actions = torch.stack(trajectories['actions']).to(device)          # shape: (N, num_edges) or similar
#     old_log_probs = torch.stack(trajectories['log_probs']).to(device)  # shape: (N, num_edges) or (N,)
#     state_values = torch.stack(trajectories['state_values']).to(device)  # shape: (N,)
#     advantages = advantages.to(device)
#     returns = returns.to(device)

#     num_samples = states.shape[0]
#     total_loss = 0.0
#     for epoch in range(epochs):
#         # Shuffle indices
#         indices = torch.randperm(num_samples)
#         for start in range(0, num_samples, batch_size):
#             end = start + batch_size
#             mini_indices = indices[start:end]

#             mini_states = states[mini_indices]
#             mini_actions = actions[mini_indices]
#             mini_old_log_probs = old_log_probs[mini_indices]
#             mini_advantages = advantages[mini_indices]
#             mini_returns = returns[mini_indices]

#             # Forward pass: Note that your actor_critic expects a graph and node features.
#             # Here we assume that mini_states are already the appropriate observations (e.g., node feature tensors).
#             # For a full implementation, you might need to repackage mini_states into a batched graph.
#             # For simplicity, assume mini_states is a tensor input to actor_critic.
#             edge_logits, state_vals = actor_critic(env.graph, mini_states)
#             dist = Categorical(logits=edge_logits)
#             new_log_probs = dist.log_prob(mini_actions)
            
#             # Calculate probability ratio
#             ratio = torch.exp(new_log_probs - mini_old_log_probs)
#             # Clipped surrogate objective
#             surrogate1 = ratio * mini_advantages
#             surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mini_advantages
#             actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
#             # Critic loss (mean squared error)
#             critic_loss = F.mse_loss(state_vals, mini_returns)
            
#             # Entropy bonus (to encourage exploration)
#             entropy_loss = -dist.entropy().mean()
            
#             loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#     avg_loss = total_loss / (epochs * (num_samples / batch_size))
#     return avg_loss


