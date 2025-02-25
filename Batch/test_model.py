import os
import csv
import logging
import torch
import dgl
import networkx as nx

# Local module imports (adjust paths as necessary)
from gymenv import EdgeColoringEnv
from actor_critic import ActorCritic, get_action
from create_graph import convert_to_undirected
from main_train_ppo  import load_graph_from_edgelist
from hyperparmeters import *
# from gymenv  import  extract_node_features

# Set up logging (if not already configured)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def evaluate_trained_model(dataset_dir, checkpoint_path, results_csv, device, max_colors, num_episodes=1):
    """
    Evaluate the trained model on each graph in the dataset.
    
    For each graph file in dataset_dir:
      - Load the graph from the edge list.
      - Convert it to undirected.
      - Create an environment instance using EdgeColoringEnv.
      - Run the trained policy (deterministically) until done=True.
      - Compute the final reward as the count of unique colors used (from graph.edata['color']).
      - Save the result (GraphFile, UniqueColorsUsed, Reward) into results_csv.
    
    Parameters:
      dataset_dir (str): Directory containing saved .edgelist files.
      checkpoint_path (str): Path to the saved trained model checkpoint.
      results_csv (str): Path to save the evaluation results CSV.
      device (torch.device): Device to run the model.
      max_colors (int): Maximum number of colors available.
      num_episodes (int): Number of evaluation runs per graph (default 1).
    
    Returns:
      None
    """
    # Hyperparameters for ActorCritic (these should match training settings)
    # num_actions = max_colors + 1
    # For in_feats, we assume the same as training (e.g., num_actions + 1)
    # in_feats = num_actions + 1
    # hidden_dim = 32
    # num_gat_heads = 4

    # Instantiate the actor-critic network and load trained parameters.
    actor_critic = ActorCritic(in_feats, hidden_dim, num_gat_heads, num_actions).to(device)
    actor_critic.load_state_dict(torch.load(checkpoint_path, map_location=device))
    actor_critic.eval()  # Set to evaluation mode

    # Open CSV file to save evaluation results.
    with open(results_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["GraphFile", "UniqueColorsUsed", "Reward"])
        
        # Iterate over each .edgelist file in the dataset directory.
        for filename in os.listdir(dataset_dir):
            if not filename.endswith(".edgelist"):
                continue
            file_path = os.path.join(dataset_dir, filename)
            logging.info(f"Evaluating graph file: {filename}")
            
            # Load the graph from file and convert to undirected.
            G_nx = load_graph_from_edgelist(file_path)
            # G_nx = convert_to_undirected(G_nx)

            # Convert NetworkX graph to DGL graph.
            graph_dgl = dgl.from_networkx(G_nx)
            
            # Create an environment instance from this graph.
            env = EdgeColoringEnv(graph_dgl, max_colors)
            
            # Run evaluation for a given number of episodes.
            total_reward = 0.0
            for ep in range(num_episodes):
                obs, _, done = env.reset()
                while not done:
                    # Extract current node features (observation).
                    node_features = env.observation
                    # Get action from the trained model.
                    action, _, _ = get_action(env.graph, node_features, actor_critic, device)
                    # Step the environment.
                    obs, reward, done, info = env.step(action)
                # Once done, final reward is computed as the number of unique colors used.
                final_colors = env.graph.edata['color'].unique().tolist()
                if 0 in final_colors:
                    final_colors.remove(0)
                episode_reward = len(final_colors)
                total_reward += episode_reward
                logging.info(f"Graph {filename} - Episode {ep+1}: Unique Colors = {final_colors}, Reward = {episode_reward}")
            
            avg_reward = total_reward / num_episodes
            # Write results to CSV.
            writer.writerow([filename, final_colors, avg_reward])
            logging.info(f"Graph {filename}: Average Reward = {avg_reward}")
    
    logging.info(f"Evaluation complete. Results saved to {results_csv}")

# Example usage:
if __name__ == "__main__":
    dataset_dir = "./test_dataset"         # Directory with saved .edgelist files
    checkpoint_path = "./checkpoints/actor_critic_update_200.pt"  # Adjust to your checkpoint file
    results_csv = "./evaluation_results.csv"
    device = torch.device("cpu")      # or "cuda"
    max_colors = max_colors  # e.g., 7

    evaluate_trained_model(dataset_dir, checkpoint_path, results_csv, device, max_colors, num_episodes=1)
