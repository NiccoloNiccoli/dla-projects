import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm


if __name__=="__main__":
    env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
    print(env.action_space)
    print(env.observation_space)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO('CnnPolicy', env, verbose=1, device=device)

    model.learn(total_timesteps=100000, log_interval=100, progress_bar=True)
    model.save('ppo_carracing')
