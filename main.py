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

# Load most recent checkpoint (by episode number in filename) if available
checkpoint_files = list(checkpoint_dir.glob("*.pth"))
def _episode_from_name(path):
    stem = path.stem  # e.g., model_ep_653_flag
    parts = stem.split("_")
    for idx, part in enumerate(parts):
        if part == "ep" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return -1
    # fallback: try last numeric chunk
    for chunk in reversed(parts):
        if chunk.isdigit():
            return int(chunk)
    return -1

checkpoint_files.sort(key=lambda p: _episode_from_name(p))
latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None
if latest_checkpoint:
    model.load_model(latest_checkpoint)
    print(f"Loaded checkpoint: {latest_checkpoint}")
else:
    print("No checkpoint found, starting fresh")

episode_log_path = Path("episode_rewards.csv")
episode_log_has_content = episode_log_path.exists() and episode_log_path.stat().st_size > 0

# Track episode rewards for later visualization (e.g., with matplotlib)
episode_rewards = []
last_episode = 0

# If continuing a previous run, load logged rewards so numbering stays consistent with the CSV
if episode_log_has_content:
    with episode_log_path.open("r") as f:
        next(f, None)  # skip header if present
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                episode_str, reward = parts[0], parts[1]
                try:
                    episode_num = int(episode_str)
                    last_episode = max(last_episode, episode_num)
                    episode_rewards.append(float(reward))
                except ValueError:
                    continue

start_episode_offset = last_episode

print(f"Starting at episode {start_episode_offset + 1}")

for i in range(NUM_OF_EPISODES):
    episode_number = start_episode_offset + i + 1
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

    # Decay epsilon once per completed episode
    model.decay_epsilon()

    print(f"Episode: {episode_number}, Reward: {episode_reward}, Epsilon: {model.epsilon:.4f}")
    episode_rewards.append(episode_reward)

    # Persist running reward history so progress can be graphed and resumed across sessions
    mode = "a" if episode_log_has_content else "w"
    with episode_log_path.open(mode) as f:
        if not episode_log_has_content:
            f.write("episode,reward,epsilon\n")
        f.write(f"{episode_number},{episode_reward},{model.epsilon}\n")
    episode_log_has_content = True

    # Save an early checkpoint on first completed episode of this session
    if episode_number == start_episode_offset + 1:
        model.save_model(checkpoint_dir / f"model_ep_{episode_number}.pth")

    if info.get("flag_get"):
        model.save_model(checkpoint_dir / f"model_ep_{episode_number}_flag.pth")

    if episode_number % 500 == 0:
        model.save_model(checkpoint_dir / f"model_ep_{episode_number}.pth")

    env.render()
env.close()
  
