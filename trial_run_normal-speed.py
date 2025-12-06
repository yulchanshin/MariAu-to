import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import torch
from pathlib import Path
import time

from wrappers import apply_wrappers
from model import Model

ENV_NAME = "SuperMarioBros-1-1-v0"
CHECKPOINT_DIR = Path("checkpoints")
NUM_EPISODES = 3

#this is a way to test the model with no randomness
def _episode_from_name(path: Path) -> int:
    parts = path.stem.split("_")
    for idx, part in enumerate(parts):
        if part == "ep" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return -1
    for chunk in reversed(parts):
        if chunk.isdigit():
            return int(chunk)
    return -1


def load_latest_model(model: Model) -> Path | None:
    ckpts = list(CHECKPOINT_DIR.glob("*.pth"))
    if not ckpts:
        return None
    ckpts.sort(key=_episode_from_name)
    latest = ckpts[-1]
    state_dict = torch.load(latest, map_location=model.online_network.device)
    model.online_network.load_state_dict(state_dict)
    model.target_network.load_state_dict(state_dict)
    return latest


def main():
    env = gym_super_mario_bros.make(
        ENV_NAME, apply_api_compatibility=True, render_mode="human"
    )
    env = JoypadSpace(env, RIGHT_ONLY)
    # Use the exact same preprocessing as training for consistent performance
    env = apply_wrappers(env)

    model = Model(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
    latest = load_latest_model(model)
    print(f"Loaded checkpoint: {latest}" if latest else "No checkpoint found; using fresh weights")

    # Evaluation only: tiny epsilon to avoid getting stuck on deterministic argmax
    model.epsilon = 0.08

    target_frame_time = 1.0 / 20.0  # real-time-ish; adjust if you want slower/faster

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.00
        last_x = 0
        stagnant_steps = 0
        patience = 30  # force a nudge if no x-position progress for this many steps
        while not done:
            step_start = time.perf_counter()
            action = model.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # If we stop making forward progress, force a right move to break the stall
            x_pos = info.get("x_pos", last_x)
            if x_pos <= last_x:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
            last_x = x_pos
            if stagnant_steps >= patience:
                action = 1  # RIGHT_ONLY: action 1 is "Right"
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                stagnant_steps = 0
                last_x = info.get("x_pos", last_x)

            state = next_state
            elapsed = time.perf_counter() - step_start
            remaining = target_frame_time - elapsed
            if remaining > 0:
                time.sleep(remaining)
        print(f"Eval episode {ep}: reward={total_reward}")

    env.close()


if __name__ == "__main__":
    main()
