# Standard Library Imports
import random

# Third-Party Imports
import gym
from gym import spaces
import torch
import numpy as np
import dgl
import networkx as nx




def compute_advantages(trajectories, gamma=0.99, lam=0.95):
    """
    Compute advantages and target returns using Generalized Advantage Estimation (GAE).

    Parameters:
        trajectories (dict): A dictionary containing the following keys:
            'rewards' (list or tensor): Sequence of rewards [r_0, r_1, ..., r_{T-1}].
            'dones' (list or tensor): Sequence of done flags [d_0, d_1, ..., d_{T-1}] 
                                      where 1 indicates the episode terminated at that step.
            'state_values' (list or tensor): Sequence of critic state value estimates [V(s_0), V(s_1), ..., V(s_{T-1})].
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter.

    Returns:
        advantages (torch.Tensor): Tensor of advantages for each time step (shape: [T]).
        returns (torch.Tensor): Tensor of target returns for each time step (advantages + state_values), shape: [T].
    """
    # Convert inputs to tensors if not already
    rewards = torch.tensor(trajectories['rewards'], dtype=torch.float32)
    dones = torch.tensor(trajectories['dones'], dtype=torch.float32)
    state_values = torch.tensor(trajectories['state_values'], dtype=torch.float32)
    
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    
    # Loop backwards over time steps
    for t in reversed(range(T)):
        # For terminal states, next state value is considered 0.
        next_value = state_values[t+1] if t < T - 1 else 0.0
        # If done[t] is 1, then (1 - done[t]) becomes 0, zeroing the future value.
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - state_values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + state_values
    return advantages, returns





# Example usage:
if __name__ == "__main__":
    # Suppose we have a trajectory of 5 steps:
    traj = {
        'rewards': [1.0, 0.5, -0.2, 0.3, 1.2],
        'dones': [0, 0, 0, 0, 1],  # Episode ends at the final step.
        'state_values': [0.8, 0.7, 0.6, 0.4, 0.0]  # Example value estimates.
    }
    advantages, returns = compute_advantages(traj, gamma=0.99, lam=0.95)
    print("Advantages:", advantages)
    print("Target Returns:", returns)






























# # =============================
# # 1. Advantage Estimation (GAE)
# # =============================
# def compute_advantages(trajectories, gamma=0.99, lam=0.95):
#     """
#     Compute advantages and target returns using Generalized Advantage Estimation (GAE).

#     Parameters:
#         trajectories (dict): Dictionary containing lists (or tensors) for:
#             - 'rewards': [r_0, r_1, ..., r_{T-1}]
#             - 'dones': [done_0, done_1, ..., done_{T-1}] (0 or 1, where 1 indicates episode termination)
#             - 'state_values': [V(s_0), V(s_1), ..., V(s_{T-1})]
#         gamma (float): Discount factor.
#         lam (float): GAE lambda parameter.

#     Returns:
#         advantages (torch.Tensor): Advantage estimates of shape (T,)
#         returns (torch.Tensor): Target returns (advantages + state_values) of shape (T,)
#     """
#     rewards = torch.tensor(trajectories['rewards'], dtype=torch.float32)
#     dones = torch.tensor(trajectories['dones'], dtype=torch.float32)
#     state_values = torch.tensor(trajectories['state_values'], dtype=torch.float32)

#     T = len(rewards)
#     advantages = torch.zeros(T, dtype=torch.float32)
#     gae = 0.0
#     # Loop backwards
#     for t in reversed(range(T)):
#         next_value = state_values[t+1] if t < T - 1 else 0.0
#         delta = rewards[t] + gamma * next_value * (1 - dones[t]) - state_values[t]
#         gae = delta + gamma * lam * (1 - dones[t]) * gae
#         advantages[t] = gae
#     returns = advantages + state_values
#     return advantages, returns
