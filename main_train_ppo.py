# # Standard Library Imports
# import random
# import os
# import random
# import logging

# # Third-Party Imports
# import gym
# from gym import spaces

# import numpy as np
# import dgl
# import networkx as nx

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim



# # Local Module Imports
# # import hyperparmeters  # If you need everything from hyperparmeters, consider aliasing or importing specific variables
# # For instance, if hyperparmeters defines MAX_COLORS and BATCH_SIZE, you could do:
# from hyperparmeters import *

# # import env_help  # Alternatively, import specific helper functions:
# # from env_help import propagate_edge_data_bidirectional, validate_action, take_action, reconcile_graph_edge_colors, extract_node_features, reset
# from env_help import (
#     propagate_edge_data_bidirectional, validate_action, take_action,
#     reconcile_graph_edge_colors, extract_node_features, reset
# )
# # import gymenv
# from gymenv import *

# # import create_graph  # Import specific functions if possible:
# from create_graph import create_bidirectional_connected_graph, convert_to_undirected

# # import actor_critic  # Import specific classes/functions from actor_critic:
# from actor_critic import ActorCritic, get_action

# # import compute_advantages
# from compute_advantages import compute_advantages

# # import collect_trajectories
# from collect_trajectories import collect_trajectories

# # import ppo_update 
# from ppo_update import ppo_update




# # # =============================
# # # 4. Main Training Loop
# # # =============================
# # def train_ppo(env, actor_critic, optimizer, num_iterations, steps_per_iteration,
# #               clip_epsilon, epochs, batch_size, device):
# #     """
# #     Main training loop for PPO.

# #     Parameters:
# #         env (EdgeColoringEnv): Gym environment.
# #         actor_critic (ActorCritic): The policy network.
# #         optimizer (torch.optim.Optimizer): Optimizer for updating actor_critic.
# #         num_iterations (int): Number of training iterations.
# #         steps_per_iteration (int): Number of transitions to collect per iteration.
# #         clip_epsilon (float): PPO clipping parameter.
# #         epochs (int): Number of epochs to update per iteration.
# #         batch_size (int): Mini-batch size for PPO update.
# #         device (torch.device): Device to run computations on.

# #     Returns:
# #         None
# #     """
# #     for iteration in range(num_iterations):
# #         print(f"\n=== Training Iteration {iteration+1} ===")
        
# #         # Collect trajectories by interacting with the environment.
# #         trajectories = collect_trajectories(env, actor_critic, steps_per_iteration, device)
        
# #         # Compute PPO update
# #         loss = ppo_update(actor_critic, optimizer, trajectories, clip_epsilon, epochs, batch_size, device)
# #         print(f"Iteration {iteration+1}: Average PPO Loss = {loss:.4f}")
        
# #         # Evaluate the current policy (optional)
# #         # You can run a few evaluation episodes and log average reward, etc.
# #         # For example:
# #         total_reward = 0
# #         num_eval_episodes = 5
# #         for _ in range(num_eval_episodes):
# #             state, done = env.reset()
# #             episode_reward = 0
# #             while not done:
# #                 node_features = env.observation
# #                 action, _, _ = get_action(env.graph, node_features, actor_critic, device)
# #                 state, reward, done, _ = env.step(action)
# #                 episode_reward += reward
# #             total_reward += episode_reward
# #         avg_reward = total_reward / num_eval_episodes
# #         print(f"Iteration {iteration+1}: Average Evaluation Reward = {avg_reward:.4f}")
        
# #         # (Optional) Save model checkpoint every few iterations.
# #         if (iteration + 1) % 10 == 0:
# #             torch.save(actor_critic.state_dict(), f"actor_critic_checkpoint_{iteration+1}.pth")
# #             print("Model checkpoint saved.")







# # # =============================
# # # 5. Main Function for Training
# # # =============================
# # if __name__ == "__main__":
# #     # Hyperparameters
# #     NUM_NODES = 6
# #     EDGE_PROB = 0.2
# #     MAX_COLORS = 7
# #     num_actions = MAX_COLORS + 1

# #     # GNN and PPO hyperparameters
# #     in_feats = num_actions + 1  # as defined in extract_node_features
# #     hidden_dim = 16
# #     num_gat_heads = 4
# #     learning_rate = 1e-3
# #     num_iterations = 50
# #     steps_per_iteration = 100  # total transitions per iteration
# #     clip_epsilon = 0.2
# #     epochs = 4
# #     batch_size = 32
# #     device = torch.device("cpu")

# #     # Create graph using your functions.
# #     graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
# #     undirected_graph = convert_to_undirected(graph_nx)
# #     graph_dgl = dgl.from_networkx(undirected_graph)

# #     # Instantiate the environment.
# #     env = EdgeColoringEnv(graph_dgl, MAX_COLORS)

# #     # Instantiate actor-critic network.
# #     actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)
# #     optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

# #     # Start PPO training loop.
# #     train_ppo(env, actor_critic, optimizer, num_iterations, steps_per_iteration,
# #               clip_epsilon, epochs, batch_size, device)



































# # def train_ppo(env, actor_critic, optimizer, num_updates, num_steps_per_update,
# #               clip_epsilon, epochs, batch_size, gamma, lam, device):
# #     """
# #     Main training loop for PPO.
    
# #     Parameters:
# #         env: Gym environment (EdgeColoringEnv instance).
# #         actor_critic: The actor-critic network.
# #         optimizer: Optimizer for updating actor_critic.
# #         num_updates: Number of PPO updates (iterations).
# #         num_steps_per_update: Number of steps (transitions) to collect per update.
# #         clip_epsilon: PPO clipping parameter.
# #         epochs: Number of epochs per update.
# #         batch_size: Mini-batch size.
# #         gamma: Discount factor.
# #         lam: GAE lambda parameter.
# #         device: torch.device.
    
# #     Returns:
# #         None
# #     """
# #     for update in range(num_updates):
# #         print(f"\n=== PPO Update {update+1}/{num_updates} ===")
# #         # Collect trajectories from the environment.
# #         trajectories = collect_trajectories(env, actor_critic, num_steps_per_update, device)
        
# #         # Compute PPO update using the collected trajectories.
# #         avg_loss = ppo_update(actor_critic, optimizer, trajectories, clip_epsilon, epochs, batch_size)
        
# #         # Optionally, compute average reward over the trajectory.
# #         avg_reward = np.mean(trajectories['rewards'])
        
# #         # Log update information.
# #         print(f"Update {update+1}: Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        
# #         # (Optional) Evaluate the current policy periodically.
# #         # e.g., run a few episodes with deterministic actions and log the performance.
        
# #     print("Training complete.")

# # # -----------------------
# # # Example usage of the PPO training loop.
# # # -----------------------
# # if __name__ == "__main__":
# #     # Hyperparameters (adjust as needed)
# #     NUM_NODES = 6
# #     EDGE_PROB = 0.2
# #     max_colors = 7  # from your hyperparameters module
# #     num_actions = max_colors + 1  # including 0 for uncolored
# #     in_feats = num_actions + 1      # example input dimension for node features (received_data + degree)
# #     hidden_dim = 16
# #     num_gat_heads = 4

# #     num_updates = 100
# #     num_steps_per_update = 200  # total steps (transitions) collected per PPO update
# #     clip_epsilon = 0.2
# #     ppo_epochs = 4
# #     batch_size = 32
# #     gamma = 0.99
# #     lam = 0.95

# #     device = torch.device("cpu")
    
# #     # Create a sample graph.
# #     graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
# #     undirected_graph = convert_to_undirected(graph_nx)
# #     graph_dgl = dgl.from_networkx(undirected_graph)
    
# #     # Instantiate environment.
# #     env = EdgeColoringEnv(graph_dgl, max_colors)
    
# #     # Reset environment to get initial observation.
# #     obs, done = env.reset()
# #     print("Initial observation (node features):")
# #     print(obs)
    
# #     # Instantiate actor-critic network.
# #     actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)
# #     optimizer = optim.Adam(actor_critic.parameters(), lr=1e-3)
    
# #     # Start PPO training loop.
# #     train_ppo(env, actor_critic, optimizer, num_updates, num_steps_per_update,
# #               clip_epsilon, ppo_epochs, batch_size, gamma, lam, device)
    






































# # Set up logging.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# # -------------------------
# # Evaluation Function
# # -------------------------
# def evaluate_policy(env_constructor, actor_critic, graph_dataset, max_colors, device, num_episodes=5):
#     """
#     Evaluate the current policy on a number of episodes (each on a different graph)
#     in a deterministic mode.
    
#     Returns the average reward over episodes.
#     """
#     total_reward = 0.0
#     for i in range(num_episodes):
#         # Sample a graph from the dataset.
#         graph = random.choice(graph_dataset)
#         # Create a new environment instance from the graph.
#         env = env_constructor(graph, max_colors)
#         obs, done = env.reset()
#         episode_reward = 0.0
#         while not done:
#             node_features = env.observation
#             action, _, _ = get_action(env.graph, node_features, actor_critic, device)
#             obs, reward, done, info = env.step(action)
#             episode_reward += reward
#         total_reward += episode_reward
#     avg_reward = total_reward / num_episodes
#     return avg_reward

# # -------------------------
# # Main PPO Training Loop
# # -------------------------
# def train_ppo_on_dataset(graph_dataset, env_constructor, actor_critic, optimizer, hyperparams, device):
#     """
#     Main training loop for PPO on a dataset of graphs.
    
#     Parameters:
#         graph_dataset: List of DGL graphs to sample from.
#         env_constructor: A function/class that creates an environment instance given a graph and max_colors.
#         actor_critic: The actor-critic network.
#         optimizer: Optimizer for updating the actor_critic network.
#         hyperparams: A dict containing hyperparameters:
#             - num_updates: Number of PPO updates.
#             - steps_per_update: Number of steps (transitions) collected per update.
#             - clip_epsilon: PPO clipping parameter.
#             - ppo_epochs: Number of epochs per update.
#             - batch_size: Mini-batch size.
#             - gamma: Discount factor.
#             - lam: GAE lambda.
#             - eval_interval: How many updates between evaluations.
#             - checkpoint_interval: How many updates between checkpointing.
#             - checkpoint_dir: Directory to save model checkpoints.
#         device: torch.device.
    
#     Returns:
#         None
#     """
#     num_updates = hyperparams['num_updates']
#     steps_per_update = hyperparams['steps_per_update']
#     clip_epsilon = hyperparams['clip_epsilon']
#     ppo_epochs = hyperparams['ppo_epochs']
#     batch_size = hyperparams['batch_size']
#     gamma = hyperparams['gamma']
#     lam = hyperparams['lam']
#     eval_interval = hyperparams['eval_interval']
#     checkpoint_interval = hyperparams['checkpoint_interval']
#     checkpoint_dir = hyperparams['checkpoint_dir']
#     max_colors = hyperparams['max_colors']
    
#     # Ensure checkpoint directory exists.
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     for update in range(1, num_updates + 1):
#         # For each update, sample one graph from the dataset.
#         graph = random.choice(graph_dataset)
#         env = env_constructor(graph, max_colors)
#         obs, reward, done = env.reset()
        
#         # Collect trajectories using the current policy.
#         trajectories = collect_trajectories(env, actor_critic, steps_per_update, device)
        
#         # Compute PPO update using the collected trajectories.
#         # avg_loss = ppo_update(actor_critic, env,  optimizer, trajectories, clip_epsilon, ppo_epochs, batch_size, device)
#         avg_loss = ppo_update(actor_critic, optimizer, trajectories, clip_epsilon, epochs, batch_size)
#         avg_reward = np.mean(trajectories['rewards'])
#         logging.info(f"Update {update}/{num_updates} - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
        
#         # Evaluate policy every eval_interval updates.
#         if update % eval_interval == 0:
#             eval_reward = evaluate_policy(env_constructor, actor_critic, graph_dataset, max_colors, device, num_episodes=5)
#             logging.info(f"Evaluation at Update {update}: Average Reward: {eval_reward:.4f}")
        
#         # Save checkpoint every checkpoint_interval updates.
#         if update % checkpoint_interval == 0:
#             checkpoint_path = os.path.join(checkpoint_dir, f"actor_critic_update_{update}.pt")
#             torch.save(actor_critic.state_dict(), checkpoint_path)
#             logging.info(f"Checkpoint saved to {checkpoint_path}")
    
#     logging.info("PPO training complete.")

# # -------------------------
# # Example Usage
# # -------------------------
# if __name__ == "__main__":
#     # Hyperparameters and settings.
#     NUM_NODES = 6
#     EDGE_PROB = 0.2
#     max_colors = 7  # Maximum colors available.
#     num_actions = max_colors + 1  # Including 0 for uncolored.
    
#     # Actor-critic hyperparameters.
#     in_feats = num_actions + 1  # Based on extracted node features (received_data + degree)
#     hidden_dim = 16
#     num_gat_heads = 4
    
#     # PPO hyperparameters.
#     hyperparams = {
#         'num_updates': 100,
#         'steps_per_update': 200,
#         'clip_epsilon': 0.2,
#         'ppo_epochs': 4,
#         'batch_size': 32,
#         'gamma': 0.99,
#         'lam': 0.95,
#         'eval_interval': 10,
#         'checkpoint_interval': 20,
#         'checkpoint_dir': "./checkpoints",
#         'max_colors': max_colors
#     }
    
#     device = torch.device("cpu")  # or "cuda" if available
    
#     # Create a dataset of graphs.
#     # For example, generate 50 graphs.
#     graph_dataset = []
#     num_graphs = 50
#     for _ in range(num_graphs):
#         graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
#         undirected_graph = convert_to_undirected(graph_nx)
#         graph_dgl = dgl.from_networkx(undirected_graph)
#         graph_dataset.append(graph_dgl)
    
#     # Instantiate the actor-critic network.
#     actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)
#     optimizer = optim.Adam(actor_critic.parameters(), lr=1e-3)
    
#     # Start PPO training on the graph dataset.
#     train_ppo_on_dataset(graph_dataset, EdgeColoringEnv, actor_critic, optimizer, hyperparams, device)










































import os
import random
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
import pyparsing

# Local imports (adjust paths as necessary)
from gymenv import *
from actor_critic import *
from compute_advantages import *
from collect_trajectories import *
from hyperparmeters import *
from env_help import *
from create_graph import *


# --- Helper functions for saving and loading graphs ---
def save_graph_as_edgelist(graph_nx, num_nodes, edge_prob, index, directory="./dataset"):
    """
    Save a NetworkX graph to an edge list file with a unique name.
    File name format: graph_num_nodes_[num_nodes]_edge_prob_[edge_prob]_number_[index].edgelist
    """
    os.makedirs(directory, exist_ok=True)
    file_name = f"graph_num_nodes_{num_nodes}_edge_prob_{edge_prob:.3f}_number_{index}.edgelist"
    file_path = os.path.join(directory, file_name)
    
    # Sort edges for consistency
    ordered_edges = sorted((min(u, v), max(u, v)) for u, v in graph_nx.edges())
    
    with open(file_path, "w") as f:
        for u, v in ordered_edges:
            f.write(f"{u} {v}\n")
    logging.info(f"Saved graph to {file_path}")
    return file_path

def load_graph_from_edgelist(file_path):
    """
    Load a graph from an edge list file into a NetworkX graph.
    """
    with open(file_path, "r") as f:
        loaded_edges = [tuple(map(int, line.strip().split())) for line in f]
    G = nx.Graph()
    G.add_edges_from(loaded_edges)
    return G


def save_training_metrics(losses, rewards, plot_filename='training_plot.png', metrics_filename='training_metrics.json'):
    # Plot training loss and reward.
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(losses, label="Avg Loss", marker='o')
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title("PPO Training Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(rewards, label="Avg Reward", color='green', marker='o')
    plt.xlabel("Update")
    plt.ylabel("Reward")
    plt.title("PPO Average Reward")
    plt.legend()

    plt.tight_layout()
    # Save the plot to a file.
    plt.savefig(plot_filename)
    plt.close()  # Close the figure

    # Save the metrics (losses and rewards) as JSON.
    metrics = {
        "losses": losses,
        "rewards": rewards
    }
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Saved training plot to '{plot_filename}' and metrics to '{metrics_filename}'.")


            # replace with actual training reward values




# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# -----------------------------
# PPO Update Function
# -----------------------------
def ppo_update(actor_critic, optimizer, trajectories, clip_epsilon, epochs, batch_size, device, graph):
    """
    Update the actor-critic network using PPO.
    
    Parameters:
      - actor_critic: the actor-critic network.
      - optimizer: optimizer for updating actor_critic.
      - trajectories: dict containing transitions collected from one episode.
          Keys: 'states', 'actions', 'rewards', 'dones', 'log_probs', 'state_values'
          * Each 'state' is a tensor of node features with shape (num_nodes, feature_dim).
          * 'actions' is a tensor of edge actions with shape (num_edges,).
      - clip_epsilon: PPO clipping parameter.
      - epochs: number of epochs to iterate over the collected data.
      - batch_size: mini-batch size (number of transitions per mini-batch).
      - device: torch.device.
      - graph: the DGL graph used in the trajectory (assumed constant for the episode).
    
    Returns:
      - avg_loss: average total loss over all updates.
    """
    # Compute advantages and target returns using GAE.
    advantages, returns = compute_advantages(trajectories, gamma=0.99, lam=0.95)
    # Convert lists to tensors if necessary.
    T = len(trajectories['states'])
    # For simplicity, we assume each state is already a tensor.
    # Also assume that the actions, log_probs, and state_values are lists of tensors.
    
    # We convert rewards, dones, log_probs, and state_values to tensors.
    rewards = torch.tensor(trajectories['rewards'], dtype=torch.float32, device=device)
    dones = torch.tensor(trajectories['dones'], dtype=torch.float32, device=device)
    old_log_probs = torch.stack(trajectories['log_probs']).to(device)   # shape: (T, num_edges)
    state_values = torch.stack(trajectories['state_values']).to(device)   # shape: (T,)
    
    # Advantages and returns are already computed as tensors.
    advantages = advantages.to(device)
    returns = returns.to(device)
    
    total_loss = 0.0
    num_updates = 0
    
    indices = torch.randperm(T)
    for epoch in range(epochs):
        # Process mini-batches
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            mini_indices = indices[start:end]
            
            batch_loss = 0.0
            for idx in mini_indices:
                # For each transition, get state, action, old_log_prob, advantage, return.
                state = trajectories['states'][idx].to(device)   # shape: (num_nodes, feature_dim)
                action = trajectories['actions'][idx].to(device)   # shape: (num_edges,)
                mini_old_log_prob = old_log_probs[idx]             # shape: (num_edges,)
                adv = advantages[idx]                              # scalar (or average over edges)
                ret = returns[idx]                                 # scalar
                
                # Forward pass: use the fixed graph from this trajectory.
                # actor_critic expects (graph, node_features)
                edge_logits, new_state_value = actor_critic(graph, state)
                # Create a distribution over edge actions.
                dist = Categorical(logits=edge_logits)  # shape: (num_edges, num_actions)
                new_log_prob = dist.log_prob(action)
                
                # Compute probability ratio.
                ratio = torch.exp(new_log_prob - mini_old_log_prob)
                # Compute surrogate losses.
                surrogate1 = ratio * adv
                surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = F.mse_loss(new_state_value, torch.tensor(ret, dtype=torch.float32, device=device))
                entropy_loss = -dist.entropy().mean()
                
                loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                num_updates += 1
            
            total_loss += batch_loss

    avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
    return avg_loss

# -----------------------------
# Main PPO Training Loop
# -----------------------------
def train_ppo(graph_dataset, num_updates, steps_per_update, clip_epsilon, ppo_epochs,
              batch_size, device, checkpoint_dir):
    """
    Main training loop for PPO over a dataset of graphs.
    
    Parameters:
      - graph_dataset: a list of DGL graphs (each graph instance).
      - num_updates: total number of PPO updates.
      - steps_per_update: number of transitions collected per update.
      - clip_epsilon: PPO clip parameter.
      - ppo_epochs: number of epochs per update.
      - batch_size: mini-batch size.
      - device: torch.device.
      - checkpoint_dir: directory for saving model checkpoints.
    
    Returns:
      None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Instantiate the actor-critic network.
    # Use hyperparameters defined in your hyperparmeters module or set here.
    # in_feats = graph_dataset[0].ndata['features'].shape[1]  # assume all graphs have same feature dim.
    
    max_colors = hyperparmeters.max_colors  # assume imported from hyperparmeters
    num_actions = max_colors + 1
    in_feats = num_actions + 1
    hidden_dim = 32  #16 to 32 generally used
    num_gat_heads = 4 # 4 to 8 generally used
    actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=5e-4)
    all_losses = []
    all_rewards = []
    # Main training loop.
    for update in range(1, num_updates+1):
        logging.info(f"=== PPO Update {update}/{num_updates} ===")
        # Sample a graph from the dataset.
        graph = random.choice(graph_dataset)
        # Create an environment instance from this graph.
        env = EdgeColoringEnv(graph, max_colors)
        obs, reward, done = env.reset()
        
        # Collect trajectories using the current policy.
        trajectories = collect_trajectories(env, actor_critic, steps_per_update, device)
        # Compute PPO update using the collected trajectories.
        avg_loss = ppo_update(actor_critic, optimizer, trajectories, clip_epsilon,
                              ppo_epochs, batch_size, device, env.graph)
        
        avg_reward = np.mean(trajectories['rewards'])
        logging.info(f"Update {update}: Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        all_losses.append(avg_loss)
        all_rewards.append(avg_reward)
        
        # Periodically evaluate the policy.
        if update % 10 == 0:
            eval_reward = evaluate_policy(EdgeColoringEnv, actor_critic, graph_dataset, max_colors, device, num_episodes=5)
            logging.info(f"Evaluation at update {update}: Average Reward = {eval_reward:.4f}")
        
        # Save checkpoint every 20 updates.
        if update % 20 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"actor_critic_update_{update}.pt")
            torch.save(actor_critic.state_dict(), ckpt_path)
            logging.info(f"Checkpoint saved to {ckpt_path}")
    
    logging.info("PPO training complete.")
    save_training_metrics(all_losses, all_rewards)



# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_policy(env_constructor, actor_critic, graph_dataset, max_colors, device, num_episodes=5):
    """
    Evaluate the current policy on a number of episodes (each on a different graph).
    
    Returns the average reward over episodes.
    """
    total_reward = 0.0
    for ep in range(num_episodes):
        graph = random.choice(graph_dataset)
        env = env_constructor(graph, max_colors)
        obs, reward, done = env.reset()
        prev_reward = reward
        episode_reward = 0.0
        while not done:
            node_features = env.observation
            action, _, _ = get_action(env.graph, node_features, actor_critic, device)
            obs, current_reward, done, info = env.step(action)
            reward_for_this_step = current_reward - prev_reward
            episode_reward += reward_for_this_step
            prev_reward = current_reward
        total_reward += episode_reward
    return total_reward / num_episodes

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Hyperparameters for training:
    
    max_colors = hyperparmeters.max_colors  # e.g., 7 from your hyperparmeters
    num_updates = 200
    steps_per_update = 1000
    clip_epsilon = 0.1
    ppo_epochs = 4
    batch_size = 32
    checkpoint_dir = "./checkpoints"
    
    device = torch.device("cpu")  # or "cuda" if available
    # Directory to save the graphs
    dataset_dir = "./dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    # Create a dataset of graphs.
    graph_dataset = []
    num_graphs = 1000
    for i in range(num_graphs):
        # Generate random parameters:
        NUM_NODES = random.randint(8, 30)
        EDGE_PROB = random.uniform(0.15, 0.2)
        
        # Create the graph (bidirectional, then convert to undirected).
        graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
        undirected_graph = convert_to_undirected(graph_nx)
        
        # Save the graph to an edge list file.
        save_graph_as_edgelist(undirected_graph, NUM_NODES, EDGE_PROB, index=i+1, directory=dataset_dir)
    
    # Now, load all graphs from the dataset directory into a list.
    graph_dataset = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".edgelist"):
            file_path = os.path.join(dataset_dir, filename)
            G_nx = load_graph_from_edgelist(file_path)
            G_dgl = dgl.from_networkx(G_nx)
            graph_dataset.append(G_dgl)
    logging.info(f"Loaded {len(graph_dataset)} graphs into the dataset.")
    
    # Start PPO training on the dataset.
    train_ppo(graph_dataset, num_updates, steps_per_update, clip_epsilon,
              ppo_epochs, batch_size, device, checkpoint_dir)

