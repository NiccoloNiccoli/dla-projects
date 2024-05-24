# Experiment 2
# Exercise 2

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import wandb
from matplotlib import animation

# Define the policy network.
class PolicyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 16)
        self.fc2 = nn.Linear(16, env.action_space.n)
        self.relu = nn.ReLU()
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s
    
class ValueNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        return s
    
def select_action(env, obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob.reshape(1)

def compute_returns(rewards, gamma=0.99):
   return np.flip(np.cumsum([gamma**(i+1)*r for (i, r) in enumerate(rewards)][::-1]), 0).copy()

def run_episode(env, policy, value_net, maxlen=1000):
    # Collect just about everything.
    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []

    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs)
        (action, log_prob) = select_action(env, obs, policy)
        value = value_net(obs)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        
        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards, values)

def reinforce(policy, value_net, env, env_render = None, gamma = 0.99, num_episodes = 10, test_every = 100, test_episodes = 10, standardize_returns = True, use_wandb = False):
    opt = optim.Adam(policy.parameters(), lr=1e-2)
    opt_value = optim.Adam(value_net.parameters(), lr=1e-2)
    running_rewards = [0.0]

    avg_rewards = []
    avg_episode_lengths = []
    
    if use_wandb:
        wandb.init(project="lab3 lunar lander value policy reinforce")


    policy.train()
    for episode in range(num_episodes):
        (observations, actions, log_probs, rewards, values) = run_episode(env, policy, value_net)
        values = torch.tensor(values, dtype=torch.float32)
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        if standardize_returns:
            returns = (returns - returns.mean()) / returns.std()
        
        targets = returns - values

        opt.zero_grad()
        opt_value.zero_grad()
        loss = (-log_probs * targets).mean()
        loss_value = F.mse_loss(returns, values)
        loss_value.requires_grad = True
        loss_value.backward()
        opt_value.step()
        loss += loss_value
        loss.backward()
        opt.step()

        if (episode+1) % test_every == 0:
            rewards = [] 
            episode_lengths = []
            policy.eval()
            value_net.eval()
            for ep in range(test_episodes):
                _, _, _, r, _ = run_episode(env, policy, value_net)
                rewards.append(np.sum(r))
                episode_lengths.append(len(r))
            print(f"Episode {episode+1}, average reward: {np.mean(rewards)}, average episode length: {np.mean(episode_lengths)}")
            print(f"Average episode length: {np.mean(episode_lengths)}")
            if use_wandb:
                wandb.log({"average_reward": np.mean(rewards), "average_episode_length": np.mean(episode_lengths)})
            avg_rewards.append(np.mean(rewards))
            avg_episode_lengths.append(np.mean(episode_lengths))
            if avg_rewards[-1] > 200:
                print("Reward greater than 200, stopping training.")
                break
            policy.train()
            value_net.train()
    policy.eval()
    value_net.eval()
    return running_rewards, avg_rewards, avg_episode_lengths

def save_frames_as_gif(frames, path='./', filename='animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=300)
    anim.save(path + filename, writer='imagemagick', fps=30)
    


if __name__ == "__main__":
    # Instantiate a rendering and a non-rendering environment.
    env_render = gym.make('LunarLander-v2', render_mode='human')
    env = gym.make('LunarLander-v2')

    # Print the observation space and action space of the environment.
    (obs, info) = env.reset()
    print("Observation: ",obs)
    print("Observation shape: ", obs.shape)
    print("Observation space: ",env.observation_space)
    print("Action space: ",env.action_space)

    policy = PolicyNet(env)
    value_net = ValueNet(env)

    # Setting parameters
    use_wandb = True
    num_episodes = 5000
    test_every = 100
    test_episodes = 10
    running_rewards, avg_rewards, avg_episode_lengths = reinforce(policy, value_net, env, env_render, num_episodes=num_episodes, test_every=test_every, test_episodes=test_episodes, use_wandb=use_wandb)

    # Plot running rewards
    plt.figure()
    plt.plot(running_rewards)
    plt.title('Running rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('lunarlander_running_rewards.png')  # Save the figure
    plt.show()

    # Plot average rewards
    plt.figure()
    plt.plot(range(test_every, num_episodes+1, test_every), avg_rewards)
    plt.title('Average rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('lunar_lander_avg_rewards.png')  # Save the figure
    plt.show()  # Display the figure

    # Plot average episode lengths
    plt.figure()
    plt.plot(range(test_every, num_episodes+1, test_every), avg_episode_lengths)
    plt.title('Average episode lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.savefig('lunarlander_avg_episode_lengths.png')  # Save the figure
    plt.show()  # Display the figure

    env_rgb = gym.make('LunarLander-v2', render_mode='rgb_array')
    observation, info = env_rgb.reset()
    frames = []
    for t in range(1000):
        frames.append(env_rgb.render())
        action, _ = select_action(env_rgb, torch.tensor(observation), policy)
        observation, _,  _, done, _ = env_rgb.step(action)
        if done:
            break
    env_rgb.close()
    save_frames_as_gif(frames, path='./', filename='lunar_lander.gif')

    if use_wandb:
        wandb.finish()