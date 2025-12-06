import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import torch
from pathlib import Path
import time

from wrappers import skip_frame
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
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
    # Eval-specific wrappers: no frame skip speedup, keep frame processing the same otherwise
    env = skip_frame(env, skip=1)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)

    model = Model(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
    latest = load_latest_model(model)
    print(f"Loaded checkpoint: {latest}" if latest else "No checkpoint found; using fresh weights")

    # Evaluation only: no exploration, no learning, no logging to CSV
    model.epsilon = 0.0

    target_frame_time = 1.0 / 60.0  # aim for ~60 FPS

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            step_start = time.perf_counter()
            action = model.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            elapsed = time.perf_counter() - step_start
            remaining = target_frame_time - elapsed
            if remaining > 0:
                time.sleep(remaining)
        print(f"Eval episode {ep}: reward={total_reward}")

    env.close()


if __name__ == "__main__":
    main()
