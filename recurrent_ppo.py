import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym


class RecurrentPPOMemory:
    def __init__(self, sequence_length):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logprobs = []
        self.returns = []
        self.advantages = []
        self.sequence_length = sequence_length
        self.hidden_states = []
    
    def append(self, state, action, reward, done, value, logprob, hidden_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.logprobs.append(logprob)
        self.hidden_states.append(hidden_state)
    
    def get_batch_indices(self, batch_size):
        trajectory_starts = np.arange(0, len(self.states), self.sequence_length)
        indices = np.random.choice(trajectory_starts, batch_size, replace=len(trajectory_starts) < batch_size)
        return indices
    
    def generate_batches(self, batch_size):
        indices = self.get_batch_indices(batch_size)
        
        batches = []
        for start_idx in indices:
            end_idx = min(start_idx + self.sequence_length, len(self.states))
            
            states_batch = self.states[start_idx:end_idx]
            actions_batch = self.actions[start_idx:end_idx]
            returns_batch = self.returns[start_idx:end_idx]
            logprobs_batch = self.logprobs[start_idx:end_idx]
            advantages_batch = self.advantages[start_idx:end_idx]
            hidden_states_batch = self.hidden_states[start_idx]
            
            batches.append((states_batch, actions_batch, returns_batch, logprobs_batch, advantages_batch, hidden_states_batch))
        
        return batches
    
    def compute_returns_and_advantages(self, next_value, gamma=0.99, gae_lambda=0.95):
        self.returns = []
        self.advantages = []
        gae = 0
        next_val = next_value
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            self.returns.insert(0, gae + self.values[t])
            self.advantages.insert(0, gae)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logprobs = []
        self.returns = []
        self.advantages = []
        self.hidden_states = []


class RecurrentActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, rnn_type='gru'):
        super(RecurrentActorCritic, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # RNN layer
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Actor (policy) network 
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.hidden_dim = hidden_dim
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def get_init_hidden(self, batch_size=1):
        if self.rnn_type == 'gru':
            return torch.zeros(1, batch_size, self.hidden_dim)
        else:  # LSTM
            return (torch.zeros(1, batch_size, self.hidden_dim),
                    torch.zeros(1, batch_size, self.hidden_dim))
    
    def forward(self, state, hidden=None, sequence_mode=False):
        features = self.feature_extractor(state)
        
        if not sequence_mode:
            features = features.unsqueeze(1)
        
        if hidden is None:
            batch_size = features.size(0)
            hidden = self.get_init_hidden(batch_size)
        
        if self.rnn_type == 'gru':
            if len(features.shape) == 2:
                if hidden.dim() == 3:
                    hidden = hidden.squeeze(1)
                rnn_out, hidden_out = self.rnn(features.unsqueeze(0), hidden.unsqueeze(1))
                rnn_out = rnn_out.squeeze(0)
            else:
                rnn_out, hidden_out = self.rnn(features, hidden)
        else:  # LSTM
            if len(features.shape) == 2:
                if hidden[0].dim() == 3:
                    h0, c0 = hidden[0].squeeze(1), hidden[1].squeeze(1)
                    rnn_out, hidden_out = self.rnn(features.unsqueeze(0), (h0.unsqueeze(1), c0.unsqueeze(1)))
                    rnn_out = rnn_out.squeeze(0)
                else:
                    rnn_out, hidden_out = self.rnn(features.unsqueeze(0), (hidden[0].unsqueeze(1), hidden[1].unsqueeze(1)))
                    rnn_out = rnn_out.squeeze(0)
            else:
                rnn_out, hidden_out = self.rnn(features, hidden)
        
        if not sequence_mode and rnn_out.dim() > 2:
            rnn_out = rnn_out.squeeze(1)
        
        action_probs = self.actor(rnn_out)
        value = self.critic(rnn_out)
        
        return action_probs, value, hidden_out
    

class RecurrentPPO:
    def __init__(self, env, hidden_dim=128, rnn_type='gru', lr=0.0005, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, epochs=4, sequence_length=128, reward_threshold=None,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.env = env
        self.reward_threshold = reward_threshold
        
        if isinstance(env.observation_space, gym.spaces.Box):
            obs_dim = env.observation_space.shape[0]
        elif isinstance(env.observation_space, gym.spaces.Dict):
            if 'observation' in env.observation_space.spaces:
                obs_dim = env.observation_space.spaces['observation'].shape[0]
                if 'mask' in env.observation_space.spaces:
                    obs_dim += 1
            else:
                obs_dim = sum(space.shape[0] if hasattr(space, 'shape') else 1 
                             for space in env.observation_space.spaces.values())
        else:
            raise ValueError(f"Unsupported observation space: {env.observation_space}")
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
        else:
            raise ValueError("Only discrete action spaces supported for now")
        
        self.policy = RecurrentActorCritic(obs_dim, hidden_dim, action_dim, rnn_type)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.sequence_length = sequence_length
        self.memory = RecurrentPPOMemory(sequence_length)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        
        self.update_count = 0
        self.reward_history = []
    
    def preprocess_obs(self, obs):
        if isinstance(obs, dict):
            if 'observation' in obs and 'mask' in obs:
                tensor_obs = np.concatenate([
                    obs['observation'].flatten(),
                    np.array([obs['mask']], dtype=np.float32)
                ])
            else:
                tensor_obs = np.concatenate([o.flatten() if hasattr(o, 'flatten') else np.array([o])
                                          for o in obs.values()])
        else:
            tensor_obs = obs

            # velocity_cartpole
            if len(tensor_obs) == 4:
                # x, x_dot, theta, theta_dot in velocity_cartpole
                
                # 1. Position
                tensor_obs[0] = tensor_obs[0] / 3.0
                
                # 2. Velocity
                tensor_obs[1] = np.clip(tensor_obs[1] / 5.0, -1.0, 1.0)
                
                # 3. Angle
                angle = tensor_obs[2]
                
                # 4. Angular velocity
                tensor_obs[3] = np.clip(tensor_obs[3] / 5.0, -1.0, 1.0)
                
                # Expand observation by adding sin and cos of the angle
                sin_theta = np.sin(angle)
                cos_theta = np.cos(angle)
                
                # Create extended observation
                tensor_obs = np.array([
                    tensor_obs[0],    # normalized position
                    tensor_obs[1],    # normalized velocity
                    sin_theta,        # sin of the angle
                    cos_theta,        # cos of the angle
                    tensor_obs[3]     # normalized angular velocity
                ], dtype=np.float32)
        
        return torch.FloatTensor(tensor_obs).to(self.device)
    
    def choose_action(self, observation, hidden_state=None):
        state = self.preprocess_obs(observation)
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            action_probs, value, hidden_state = self.policy(state, hidden_state)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), value.item(), log_prob.item(), hidden_state
    
    def learn(self, total_timesteps, max_ep_len=1000):
        timestep = 0
        episode = 0
        best_reward = float('-inf')
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        while timestep < total_timesteps:
            obs, _ = self.env.reset()
            hidden_state = self.policy.get_init_hidden()
            hidden_state = tuple(h.to(self.device) for h in hidden_state) if isinstance(hidden_state, tuple) else hidden_state.to(self.device)
            
            done = False
            episode_reward = 0
            episode_timesteps = 0
            
            while not done and episode_timesteps < max_ep_len:
                action, value, log_prob, next_hidden = self.choose_action(obs, hidden_state)
                
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.memory.append(
                    self.preprocess_obs(obs).cpu().numpy(),
                    action,
                    reward,
                    done,
                    value,
                    log_prob,
                    hidden_state if isinstance(hidden_state, torch.Tensor) else hidden_state[0].cpu().numpy()
                )
                
                obs = next_obs
                hidden_state = next_hidden
                episode_reward += reward
                timestep += 1
                episode_timesteps += 1
                
                if done or episode_timesteps >= max_ep_len or timestep % self.sequence_length == 0:
                    if not done:
                        with torch.no_grad():
                            _, next_value, _, _ = self.choose_action(obs, hidden_state)
                    else:
                        next_value = 0.0
                    
                    self.memory.compute_returns_and_advantages(next_value, self.gamma, self.gae_lambda)
                    
                    if len(self.memory.states) >= self.sequence_length:
                        self.update()
                        self.memory.clear()
                        
                        progress = min(1.0, timestep / total_timesteps)
                        new_lr = initial_lr * (1.0 - progress)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
            
            episode += 1
            self.reward_history.append(episode_reward)
            avg_reward = np.mean(self.reward_history[-50:]) if len(self.reward_history) >= 50 else np.mean(self.reward_history)
            
            print(f"Episode {episode}, reward: {episode_reward:.2f}, avg_reward (last 50): {avg_reward:.2f}, timesteps: {timestep}/{total_timesteps}")
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(self.policy.state_dict(), "best_recurrent_ppo.pt")
                print(f"New best model saved with reward: {best_reward}")
                
                if best_reward >= 500.0 and len(self.reward_history) >= 5 and np.mean(self.reward_history[-5:]) >= 490.0:
                    print(f"🎉 Environment solved! Stopping training early at timestep {timestep}")
                    break
        
        print(f"Training complete. Best reward: {best_reward}")
        return self.policy
    
    def update(self):
        self.update_count += 1
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.memory.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.memory.logprobs)).to(self.device)
        returns = torch.FloatTensor(np.array(self.memory.returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(self.memory.advantages)).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            for i in range(0, len(states), self.sequence_length):
                end_idx = min(i + self.sequence_length, len(states))
                if end_idx - i < 4:
                    continue
                    
                seq_states = states[i:end_idx]
                seq_actions = actions[i:end_idx]
                seq_old_logprobs = old_logprobs[i:end_idx]
                seq_returns = returns[i:end_idx]
                seq_advantages = advantages[i:end_idx]
                
                hidden = self.policy.get_init_hidden(batch_size=1)
                hidden = tuple(h.to(self.device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(self.device)
                
                action_probs, values, _ = self.policy(seq_states, hidden, sequence_mode=True)
                values = values.squeeze(-1)
                
                dist = Categorical(action_probs)
                new_logprobs = dist.log_prob(seq_actions)
                ratio = torch.exp(new_logprobs - seq_old_logprobs)
                
                surr1 = ratio * seq_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * seq_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * ((values - seq_returns) ** 2).mean()
                
                entropy = dist.entropy().mean()
                
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                if self.update_count % 100 == 0:
                    with torch.no_grad():
                        approx_kl = ((seq_old_logprobs - new_logprobs) ** 2).mean().item()
                        policy_entropy = entropy.item()
                        clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                        
                    print(f"Update {self.update_count} - Loss: {loss.item():.4f}, "
                          f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, "
                          f"Entropy: {policy_entropy:.4f}, KL: {approx_kl:.4f}, Clip fraction: {clipfrac:.4f}")
    
    def collect_trajectories(self, num_trajectories, render=False, rtg_gamma=1.0):
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_rtgs = []
        all_timesteps = []
        
        for traj_idx in range(num_trajectories):
            states, actions, rewards, dones, rtgs, timesteps = [], [], [], [], [], []
            
            if render:
                env = gym.make(self.env.unwrapped.spec.id, render_mode="human")
            else:
                env = self.env
                
            obs, _ = env.reset()
            hidden_state = self.policy.get_init_hidden()
            hidden_state = tuple(h.to(self.device) for h in hidden_state) if isinstance(hidden_state, tuple) else hidden_state.to(self.device)
            
            done = False
            episode_reward = 0
            t = 0
            
            while not done:
                raw_obs = obs
                if isinstance(raw_obs, dict):
                    raw_obs = raw_obs['observation'] if 'observation' in raw_obs else np.array(list(raw_obs.values()))
                
                states.append(raw_obs)
                timesteps.append(t)
                
                action, _, _, next_hidden = self.choose_action(obs, hidden_state)
                actions.append(action)
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                rewards.append(reward)
                dones.append(done)
                episode_reward += reward
                
                obs = next_obs
                hidden_state = next_hidden
                t += 1
            
            returns = 0
            for r in reversed(rewards):
                ##returns = r + self.gamma * returns
                returns = r + rtg_gamma * returns
                rtgs.append(returns)
            rtgs.reverse()
            
            all_states.append(np.array(states))
            all_actions.append(np.array(actions))
            all_rewards.append(np.array(rewards))
            all_dones.append(np.array(dones))
            all_rtgs.append(np.array(rtgs))
            all_timesteps.append(np.array(timesteps))
            
            print(f"Trajectory {traj_idx+1}/{num_trajectories}, reward: {episode_reward}")
            
            if render:
                env.close()
        
        return all_states, all_actions, all_rewards, all_dones, all_rtgs, all_timesteps
    
    def save_trajectories(self, output_dir, num_trajectories=1000, rtg_gamma=1.0):
        """
        Collect and save trajectories for training a Decision Transformer.
        Only save trajectories where total reward >= reward_threshold.
        If reward_threshold is None, all trajectories are saved.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        attempt_count = 0
        max_attempts = num_trajectories * 5 #* 3
        
        while saved_count < num_trajectories and attempt_count < max_attempts:
            states, actions, rewards, dones, rtgs, timesteps = self.collect_trajectories(1, rtg_gamma=rtg_gamma)
            
            total_reward = sum(rewards[0])
            meets_threshold = True
            if hasattr(self, 'reward_threshold') and self.reward_threshold is not None:
                meets_threshold = total_reward >= self.reward_threshold
            
            if meets_threshold:
                traj_data = {
                    'obs': states[0],
                    'action': actions[0],
                    'reward': rewards[0],
                    'done': dones[0],
                    'rtg': rtgs[0],
                    'timesteps': timesteps[0],
                    'rtg_gamma': np.array(rtg_gamma),
                    'ppo_gamma': np.array(self.gamma),
                }
                
                file_path = f'{output_dir}/train_data_{saved_count}.npz'
                np.savez(file_path, **traj_data)
                saved_count += 1
                print(f"Saved trajectory {saved_count}/{num_trajectories} with reward {total_reward:.2f}")
            
            attempt_count += 1
        
        if saved_count < num_trajectories:
            print(f"Warning: Only collected {saved_count}/{num_trajectories} trajectories " 
                  f"meeting the reward threshold after {attempt_count} attempts")
        else:
            print(f"Successfully saved {num_trajectories} trajectories to {output_dir}")