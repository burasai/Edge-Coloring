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
import dgl
import logging
# Local imports (adjust paths as necessary)
from gymenv import *
from actor_critic import *
from compute_advantages import *
from collect_trajectories import *
from hyperparmeters import *
# from env_help import *
from create_graph import *








# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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
# PPO Update Function (Device Compatible)
# -----------------------------
###############################################
# PPO Update Function (Flattened Version)
###############################################
def flatten_trajectories(trajectories, batch_mode=False):
    """
    Flatten trajectory lists into a single tensor per key.
    
    For keys that are stored as lists of values (tensors or numbers) with shape [B] per time step,
    we convert each element to a tensor and then concatenate along dim=0 to obtain a tensor of shape [T*B].
    For 'states' and 'actions':
      - In single-env mode (batch_mode False), we simply stack.
      - In batched mode (batch_mode True), we assume each state is a tensor of shape [N, feat]
        and each action is a tensor of shape [E]. We then replicate each state (and action) B times
        (where B is the batch size) and then concatenate.
    """
    flat_traj = {}
    keys_to_flatten = ['rewards', 'dones', 'log_probs', 'state_values']
    for key in keys_to_flatten:
        new_list = []
        for item in trajectories[key]:
            if not isinstance(item, torch.Tensor):
                item = torch.tensor(item, dtype=torch.float32)
            new_list.append(item)
        if batch_mode:
            flat_traj[key] = torch.cat(new_list, dim=0)  # shape: [T*B]
        else:
            flat_traj[key] = torch.stack(new_list)  # shape: [T]
    
    if batch_mode:
        B = trajectories['rewards'][0].shape[0] if isinstance(trajectories['rewards'][0], torch.Tensor) else 1
        states_list = []
        for state in trajectories['states']:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            replicated = state.unsqueeze(0).repeat(B, 1, 1)  # [B, N, feat]
            states_list.append(replicated)
        flat_traj['states'] = torch.cat(states_list, dim=0)  # [T*B, N, feat]
        
        actions_list = []
        for action in trajectories['actions']:
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.int64)
            replicated = action.unsqueeze(0).repeat(B, 1)  # [B, E]
            actions_list.append(replicated)
        flat_traj['actions'] = torch.cat(actions_list, dim=0)  # [T*B, E]
    else:
        flat_traj['states'] = torch.stack(trajectories['states'])
        flat_traj['actions'] = torch.stack(trajectories['actions'])
    
    return flat_traj

def ppo_update(actor_critic, optimizer, trajectories, clip_epsilon, epochs, mini_batch_size, device, graph):
    """
    Update the actor-critic network using PPO.

    This function computes advantages and returns using GAE, then flattens the trajectories (if batched)
    so that each transition is treated independently. It filters out any transitions where the action tensor
    is empty, then performs mini-batch updates.
    
    Parameters:
      - actor_critic: The actor-critic network.
      - optimizer: Optimizer for updating actor_critic.
      - trajectories: Dictionary containing transitions from one episode.
          Keys: 'states', 'actions', 'rewards', 'dones', 'log_probs', 'state_values'
      - clip_epsilon: PPO clipping parameter.
      - epochs: Number of epochs over the collected data.
      - mini_batch_size: Mini-batch size.
      - device: Torch device.
      - graph: The DGL graph used in the trajectory (assumed constant).
    
    Returns:
      - avg_loss: Average loss over all updates.
    """
    graph = graph.to(device)
    advantages, returns = compute_advantages(trajectories, gamma=0.99, lam=0.95, device=device)
    
    # Determine if batched: rewards are 2D => shape [T, B]
    batch_mode = advantages.dim() == 2
    if batch_mode:
        T, B = advantages.shape
        advantages = advantages.view(T * B)
        returns = returns.view(T * B)
    else:
        advantages = advantages.view(-1)
        returns = returns.view(-1)
    
    flat_traj = flatten_trajectories(trajectories, batch_mode=batch_mode)
    old_log_probs = flat_traj['log_probs']
    state_values = flat_traj['state_values']
    if state_values.dim() == 2:
        state_values = state_values.view(-1)
    else:
        state_values = state_values.view(-1)
    
    # Filter out transitions with empty actions.
    valid_indices = [i for i in range(flat_traj['actions'].shape[0]) if flat_traj['actions'][i].numel() > 0]
    if len(valid_indices) == 0:
        return 0.0  # If no valid transitions, return zero loss.
    
    valid_indices = torch.tensor(valid_indices, device=device)
    # Randomize valid indices.
    indices = valid_indices[torch.randperm(valid_indices.numel())]
    
    N = indices.shape[0]
    total_loss = 0.0
    num_updates = 0
    
    for epoch in range(epochs):
        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            mini_indices = indices[start:end]
            batch_loss = 0.0
            for idx in mini_indices:
                # Retrieve transition.
                state = flat_traj['states'][idx]      # shape: [N_nodes, feat]
                action = flat_traj['actions'][idx]      # shape: [num_edges]
                mini_old_log_prob = old_log_probs[idx]
                adv = advantages[idx]
                ret = returns[idx]
                
                # Forward pass.
                edge_logits, new_state_value = actor_critic(graph, state)
                dist = torch.distributions.Categorical(logits=edge_logits)
                new_log_prob = dist.log_prob(action)
                
                ratio = torch.exp(new_log_prob - mini_old_log_prob)
                surrogate1 = ratio * adv
                surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = torch.nn.functional.mse_loss(new_state_value, torch.tensor(ret, dtype=torch.float32, device=device))
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



###############################################
# PPO Training Loop
###############################################

def train_ppo(graph_dataset, num_updates, steps_per_update, clip_epsilon, ppo_epochs,
              batch_size, device, checkpoint_dir, env_batch_size):
    """
    Main training loop for PPO over a dataset of graphs.

    This function supports either a single-graph or a batched environment (using dgl.batch).
    For batched mode, env_batch_size determines how many graphs are batched together.

    Parameters:
      - graph_dataset: List of DGL graphs.
      - num_updates: Total number of PPO updates.
      - steps_per_update: Number of transitions to collect per update.
      - clip_epsilon: PPO clipping parameter.
      - ppo_epochs: Number of epochs per update.
      - batch_size: Mini-batch size for PPO update.
      - device: Torch device.
      - checkpoint_dir: Directory for saving model checkpoints.
      - env_batch_size: Number of graphs per environment batch.

    Returns:
      None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Hyperparameters (imported from hyperparmeters)
    max_colors_val = max_colors  # e.g., 7
    num_actions = max_colors_val + 1
    in_feats = num_actions + 1   # As defined during feature extraction.
    hidden_dim = 32
    num_gat_heads = 4
    
    print(f"Using env batch size: {env_batch_size}")
    
    # Instantiate actor-critic network.
    actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=5e-4)
    
    all_losses = []
    all_rewards = []
    
    for update in range(1, num_updates + 1):
        logging.info(f"=== PPO Update {update}/{num_updates} ===")
        # Sample a set of graphs according to env_batch_size.
        sampled_graphs = random.sample(graph_dataset, env_batch_size)
        # Create a batched environment if env_batch_size > 1.
        if env_batch_size > 1:
            env = EdgeColoringEnv(sampled_graphs, max_colors, device=device)
        else:
            env = EdgeColoringEnv(sampled_graphs[0], max_colors, device=device)
        
        obs, reward, done = env.reset()
        print(f"\nRunning PPO update {update} on env with device {env.device}")
        
        trajectories = collect_trajectories(env, actor_critic, steps_per_update, device)
        avg_loss = ppo_update(actor_critic, optimizer, trajectories, clip_epsilon,
                              ppo_epochs, batch_size, device, env.graph)
        
        # For reward, if batched, take the mean over the batch.
        if isinstance(reward, torch.Tensor):
            avg_reward = torch.mean(torch.stack(trajectories['rewards'])).item()
        else:
            avg_reward = np.mean(trajectories['rewards'])
        logging.info(f"Update {update}: Avg Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        all_losses.append(avg_loss)
        all_rewards.append(avg_reward)
        
        # Periodic evaluation.
        if update % 10 == 0:
            eval_reward = evaluate_policy(EdgeColoringEnv, actor_critic, graph_dataset, max_colors_val, device, num_episodes=5)
            logging.info(f"Evaluation at update {update}: Average Reward = {eval_reward:.4f}")
        
        # Save checkpoint every 20 updates.
        if update % 20 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"actor_critic_update_{update}.pt")
            torch.save(actor_critic.state_dict(), ckpt_path)
            logging.info(f"Checkpoint saved to {ckpt_path}")
    
    logging.info("PPO training complete.")
    
    # Plot and save training metrics.
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(all_losses, marker='o', label="Avg Loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title("PPO Training Loss")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(all_rewards, marker='o', label="Avg Reward", color='green')
    plt.xlabel("Update")
    plt.ylabel("Reward")
    plt.title("PPO Average Reward")
    plt.legend()
    
    plt.tight_layout()
    plot_filename = os.path.join(checkpoint_dir, "training_plot.png")
    plt.savefig(plot_filename)
    logging.info(f"Training plot saved to {plot_filename}")
    
    metrics = {"losses": all_losses, "rewards": all_rewards}
    metrics_filename = os.path.join(checkpoint_dir, "training_metrics.json")
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Training metrics saved to {metrics_filename}")

###############################################
# Evaluation Function
###############################################
def evaluate_policy(env_constructor, actor_critic, graph_dataset, max_colors, device, num_episodes=5):
    """
    Evaluate the current policy over a number of episodes (each using a different graph).

    For each episode:
      - A random graph is selected.
      - An environment is created from the graph.
      - The environment is reset, yielding an initial cumulative reward.
      - The actor-critic selects actions until the episode terminates.
      - Instead of summing step-by-step incremental rewards, the final cumulative reward is used
        as the performance metric for that episode.

    Returns:
        float: The average final cumulative reward over the evaluated episodes.
    """
    total_final_reward = 0.0
    for ep in range(num_episodes):
        # Sample a random graph from the dataset.
        graph = random.choice(graph_dataset)
        # Create an environment instance.
        env = env_constructor(graph, max_colors, device=device)
        # Reset the environment.
        obs, initial_reward, done = env.reset()
        # Run the episode until termination.
        while not done:
            node_features = env.observation.to(device)
            action, _, _ = get_action(env.graph, node_features, actor_critic, device)
            obs, cumulative_reward, done, info = env.step(action)
        # Use the final cumulative reward as the episode's performance.
        episode_final_reward = cumulative_reward
        total_final_reward += episode_final_reward
        print(f"Episode {ep+1}: Final Cumulative Reward = {episode_final_reward}")
    avg_reward = total_final_reward / num_episodes
    print(f"Average Final Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

###############################################
# End of Module
###############################################



    



if __name__ == "__main__":
    # Hyperparameters for training:
    max_colors_val = max_colors    # e.g., 7
    num_updates = 100
    steps_per_update = 1000
    clip_epsilon = 0.1
    ppo_epochs = 4
    batch_size = 32           # mini-batch size for PPO update
    env_batch_size = 25       # number of graphs in each batched environment
    checkpoint_dir = "./checkpoints"
    
    # Set device: use "cuda" if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directory for saving graphs.
    dataset_dir = "./test_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create a dataset of graphs.
    graph_dataset = []
    num_graphs = 1000
    for i in range(num_graphs):
        # Generate random graph parameters.
        NUM_NODES = random.randint(6,8)
        EDGE_PROB = random.uniform(0.18, 0.2)
        
        # Create a bidirectional graph and convert it to an undirected graph.
        graph_nx = create_bidirectional_connected_graph(NUM_NODES, EDGE_PROB)
        undirected_graph = convert_to_undirected(graph_nx)
        
        # Save the graph as an edgelist file.
        save_graph_as_edgelist(undirected_graph, NUM_NODES, EDGE_PROB, index=i+1, directory=dataset_dir)
    
    # Load all graphs from the dataset directory.
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".edgelist"):
            file_path = os.path.join(dataset_dir, filename)
            G_nx = load_graph_from_edgelist(file_path)
            G_dgl = dgl.from_networkx(G_nx)
            graph_dataset.append(G_dgl)
    logging.info(f"Loaded {len(graph_dataset)} graphs into the dataset.")
    
    # Save the training parameters to a JSON file in the checkpoint directory.
    os.makedirs(checkpoint_dir, exist_ok=True)
    params = {
        "max_colors": max_colors_val,
        "num_updates": num_updates,
        "steps_per_update": steps_per_update,
        "clip_epsilon": clip_epsilon,
        "ppo_epochs": ppo_epochs,
        "batch_size": batch_size,
        "env_batch_size": env_batch_size,
        "in_feats": in_feats,
        "hidden_dim": hidden_dim,
        "num_gat_heads": num_gat_heads,
        "num_actions": num_actions,
        "device": str(device)
    }
    params_filename = os.path.join(checkpoint_dir, "training_parameters.json")
    with open(params_filename, "w") as f:
        json.dump(params, f, indent=4)
    logging.info(f"Training parameters saved to {params_filename}")
    
    # Start PPO training on the dataset using the batched environment.
    train_ppo(graph_dataset, num_updates, steps_per_update, clip_epsilon, ppo_epochs,
              batch_size, device, checkpoint_dir, env_batch_size)
    
    logging.info("Training complete.")
