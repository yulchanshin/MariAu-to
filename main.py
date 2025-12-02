import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from wrappers import apply_wrappers
from model import Model
import os
from pathlib import Path


ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_OF_EPISODES = 50000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

checkpoint_dir = Path("checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

model = Model(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    episode_reward = 0.0

    while not done:
        action = model.choose_action(state)

        new_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        model.store_in_memory(state, action, reward, new_state, done)

        model.learn()

        state = new_state
        episode_reward += reward

    print(f"Episode: {i + 1}, Reward: {episode_reward}")

    if i == 1:
        model.save_model(checkpoint_dir / f"model_ep_{i}.pth")

    if info.get("flag_get"):
        model.save_model(checkpoint_dir / f"model_ep_{i + 1}_flag.pth")

    if (i + 1) % 1000 == 0:
        model.save_model(checkpoint_dir / f"model_ep_{i + 1}.pth")

    env.render()
env.close()
  
