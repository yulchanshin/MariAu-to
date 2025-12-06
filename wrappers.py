import numpy as np
import gymnasium as gym
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class skip_frame(Wrapper):
    # skip is the amount of frames skipped
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return next_state, total_reward, done, trunc, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.frames_log = [state.copy()]
        self.actions_log = [0]

        return state, info


def apply_wrappers(env):
    env = skip_frame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env
