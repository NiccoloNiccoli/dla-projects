import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import imageio
from itertools import count
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_training_envs', type=int, default=12)
    parser.add_argument('--num_eval_envs', type=int, default=12)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--best-model-save-path', type=str, default='./DRL/marioBros')
    parser.add_argument('--log-path', type=str, default='./logs/')
    parser.add_argument('--model-path', type=str, default='.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gif-path', type=str, default='.')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    env_id = 'ALE/MarioBros-v5'
    num_training_envs = args.num_training_envs
    num_eval_envs = args.num_eval_envs
    timesteps = args.timesteps
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the training and evaluation environments
    vec_env = make_vec_env(env_id, n_envs=num_training_envs, seed = args.seed, vec_env_cls = SubprocVecEnv)
    eval_env = make_vec_env(env_id, n_envs = num_eval_envs, seed = args.seed, vec_env_cls = SubprocVecEnv)
    eval_callback = EvalCallback(eval_env, best_model_save_path = args.best_model_save_path, log_path = args.log_path, eval_freq = max(100_000 // num_eval_envs, 1), n_eval_episodes=12, deterministic = True, render = False)

    # Create the model
    model = PPO('CnnPolicy', vec_env, verbose=1, device=device)

    # Train the model
    model.learn(total_timesteps=timesteps, log_interval=100, progress_bar=True, callback=eval_callback)

    model.save(f'{args.model_path}/ppo_marioBros_{timesteps}')

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')

    # Saving the video and a gif
    env = model.get_env()
    test_env = make_vec_env(env_id, n_envs = 1, seed = 0) 

    images = []
    obs = test_env.reset()
    img = test_env.render(mode='rgb_array')
    done = False
    for t in count():
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, done, _ = test_env.step(action)
        img = test_env.render(mode='rgb_array')
        if done:
            break
    
    imageio.mimsave(f'{args.gif_path}/ppo_marioBros_{timesteps}.gif', [np.array(img) for i, img in enumerate(images)], fps=30)