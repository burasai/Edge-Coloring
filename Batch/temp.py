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
