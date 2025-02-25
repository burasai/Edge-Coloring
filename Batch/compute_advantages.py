# Standard Library Imports
import random

# Third-Party Imports
import gym
from gym import spaces
import torch
import numpy as np
import dgl
import networkx as nx
# from collect_trajectories import *


###############################################
# Compute Advantages Function
###############################################

# def compute_advantages(trajectories, gamma=0.99, lam=0.95, device=None):
#     """
#     Compute advantages and target returns using Generalized Advantage Estimation (GAE).

#     Parameters:
#         trajectories (dict): A dictionary containing:
#             'rewards': List of rewards from each time step.
#                        For a batched environment, each element is a tensor of shape [B] (or a number).
#             'dones': List of done flags from each time step.
#                      For a batched environment, each element is a tensor of shape [B] (or a number, 1 for done).
#             'state_values': List of state value estimates from the critic.
#                            Same shape as rewards.
#         gamma (float): Discount factor.
#         lam (float): GAE lambda parameter.
#         device (torch.device): Device for computation.

#     Returns:
#         advantages (torch.Tensor): Advantages per time step.
#             For batched env, shape [T, B]; for single env, shape [T].
#         returns (torch.Tensor): Target returns (advantages + state_values), same shape as advantages.
#     """
#     if device is None:
#         device = torch.device("cpu")
    
#     # Convert each element to a tensor if necessary.
#     rewards = torch.stack([
#         r if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32, device=device)
#         for r in trajectories['rewards']
#     ]).to(device)
    
#     dones = torch.stack([
#         d if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=torch.float32, device=device)
#         for d in trajectories['dones']
#     ]).to(device)
    
#     state_values = torch.stack([
#         v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=device)
#         for v in trajectories['state_values']
#     ]).to(device)
    
#     T = rewards.shape[0]
    
#     # Check if we are in single env (1D rewards) or batched env (2D rewards).
#     if rewards.dim() == 1:
#         advantages = torch.zeros(T, dtype=torch.float32, device=device)
#         gae = 0.0
#         for t in reversed(range(T)):
#             next_value = state_values[t+1] if t < T - 1 else 0.0
#             delta = rewards[t] + gamma * next_value * (1 - dones[t]) - state_values[t]
#             gae = delta + gamma * lam * (1 - dones[t]) * gae
#             advantages[t] = gae
#         returns = advantages + state_values
#     else:
#         # Batched mode: rewards shape [T, B].
#         T, B = rewards.shape
#         advantages = torch.zeros(T, B, dtype=torch.float32, device=device)
#         gae = torch.zeros(B, dtype=torch.float32, device=device)
#         for t in reversed(range(T)):
#             next_value = state_values[t+1] if t < T - 1 else torch.zeros(B, dtype=torch.float32, device=device)
#             delta = rewards[t] + gamma * next_value * (1 - dones[t]) - state_values[t]
#             gae = delta + gamma * lam * (1 - dones[t]) * gae
#             advantages[t] = gae
#         returns = advantages + state_values

#     return advantages, returns



















###############################################
# Compute Advantages Function
###############################################

def compute_advantages(trajectories, gamma=0.99, lam=0.95, device=None):
    """
    Compute advantages and target returns using Generalized Advantage Estimation (GAE).

    For a batched environment, this function expects that each time step's reward, done, and state_value
    are tensors of shape [B]. Stacking over T steps produces tensors of shape [T, B]. We then compute
    advantages per environment (along T) and return advantages and target returns of shape [T, B].
    For a single environment, these are 1D tensors of shape [T].

    Parameters:
        trajectories (dict): Dictionary with keys:
            'rewards': List of rewards (each element is either a scalar or a tensor of shape [B]).
            'dones': List of done flags (each element is either a scalar or tensor of shape [B]).
            'state_values': List of critic value estimates (same shape as rewards).
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter.
        device (torch.device): Device for computation.

    Returns:
        advantages (torch.Tensor): Advantages per time step ([T] for single env or [T, B] for batched).
        returns (torch.Tensor): Target returns (advantages + state_values), same shape.
    """
    if device is None:
        device = torch.device("cpu")
    
    rewards = torch.stack([
        r if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32, device=device)
        for r in trajectories['rewards']
    ]).to(device)
    
    dones = torch.stack([
        d if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=torch.float32, device=device)
        for d in trajectories['dones']
    ]).to(device)
    
    state_values = torch.stack([
        v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=device)
        for v in trajectories['state_values']
    ]).to(device)
    
    T = rewards.shape[0]
    if rewards.dim() == 1:
        # Single environment.
        advantages = torch.zeros(T, dtype=torch.float32, device=device)
        gae = 0.0
        for t in reversed(range(T)):
            next_value = state_values[t+1] if t < T - 1 else 0.0
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - state_values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + state_values
    else:
        # Batched: shape [T, B]
        T, B = rewards.shape
        advantages = torch.zeros(T, B, dtype=torch.float32, device=device)
        gae = torch.zeros(B, dtype=torch.float32, device=device)
        for t in reversed(range(T)):
            next_value = state_values[t+1] if t < T - 1 else torch.zeros(B, dtype=torch.float32, device=device)
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - state_values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + state_values

    return advantages, returns

























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
