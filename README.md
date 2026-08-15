<h1 align="center">MariAu-to</h1>

<p align="center">
  A deep Q-network that learns to play <em>Super Mario Bros</em> from raw pixels.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.1-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

MariAu-to trains a convolutional Q-network on `SuperMarioBros-1-1` using nothing but the
game screen. There is no hand written feature extraction, no access to the emulator's
internal state during action selection, and no reward shaping beyond what the environment
provides. The agent sees four stacked 84x84 grayscale frames and picks one of five
right-facing button combinations.

The current checkpoint is the result of roughly 36,900 episodes of training and clears
world 1-1 regularly.

## Quickstart

```bash
git clone https://github.com/yulchanshin/MariAu-to.git
cd MariAu-to

python -m venv venv
source venv/bin/activate     # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Train:

```bash
python main.py
```

Training resumes from the highest numbered checkpoint in `checkpoints/` and appends to
`episode_rewards.csv`, so a run can be stopped and restarted without losing progress.

Watch a trained agent at roughly real time speed:

```bash
python trial_run_normal-speed.py
```

`trial_run_normal-speed.py` loads the latest checkpoint, throttles to 20 steps per second,
and runs three evaluation episodes.

Python 3.10 or 3.11 is required. `nes-py` does not build cleanly on newer versions.

## How it works

### The problem

Each frame of the game is a state $s$, each button combination is an action $a$, and the
environment returns a scalar reward $r$. `gym-super-mario-bros` defines that reward as

$$
r = \Delta x + c + d
$$

where $\Delta x$ is how far Mario moved right since the last frame, $c$ is a small negative
clock penalty that discourages standing still, and $d$ is $-15$ on death. The sum is clipped
to $[-15, 15]$. Maximizing cumulative reward therefore means moving right quickly without dying.

The goal is the optimal action-value function $Q^{\ast}(s, a)$, the expected discounted return
from taking action $a$ in state $s$ and acting optimally afterward. It satisfies the Bellman
optimality equation:

$$
Q^{\ast}(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^{\ast}(s', a') \right]
$$

Tabular Q-learning cannot be used here because the state space is the set of all possible
screens. Instead a neural network $Q_\theta$ approximates $Q^{\ast}$.

### Observation preprocessing

Four wrappers sit between the emulator and the agent (`wrappers.py`):

| Wrapper | Effect | Why |
| --- | --- | --- |
| `skip_frame(4)` | Repeats an action for 4 frames, returns the summed reward | Consecutive frames carry almost no new information, and this cuts the decision rate by 4x |
| `ResizeObservation(84)` | Downsamples to 84x84 | Standard DQN input size, keeps the conv stack small |
| `GrayScaleObservation` | Drops color | Color is not needed for the control task |
| `FrameStack(4)` | Concatenates the last 4 observations | Makes the state Markov |

The frame stack matters more than it looks. A single frame does not tell you whether Mario
is rising or falling, so the process is not Markov and Q-learning has no fixed point to
converge to. Stacking four frames restores enough state that velocity is recoverable.

The final observation is a $4 \times 84 \times 84$ tensor covering 16 game frames.

### Network

`ModelNN.py` uses the standard Atari DQN architecture:

```
Input   4 x 84 x 84
Conv    32 filters, 8x8, stride 4  ->  32 x 20 x 20   ReLU
Conv    64 filters, 4x4, stride 2  ->  64 x  9 x  9   ReLU
Conv    64 filters, 3x3, stride 1  ->  64 x  7 x  7   ReLU
Flatten                            ->  3136
Linear  3136 -> 512                                   ReLU
Linear  512  -> 5
```

Each spatial dimension follows $\lfloor (n - k) / s \rfloor + 1$, giving 20, then 9, then 7.
The five outputs are the Q-values for the `RIGHT_ONLY` action set. The network is never told
what an action means; the mapping is learned from reward alone.

### The learning rule

Two copies of the network are kept: an online network $Q_\theta$ that is updated every step,
and a target network $Q_{\theta^-}$ that is a frozen snapshot of the online weights. For a
minibatch of transitions $(s, a, r, s', \text{done})$ sampled from replay, the regression
target is

$$
y = r + \gamma \, (1 - \text{done}) \max_{a'} Q_{\theta^-}(s', a')
$$

and the loss is the mean squared error against the online network's prediction:

$$
L(\theta) = \left( y - Q_\theta(s, a) \right)^2
$$

The $(1 - \text{done})$ factor zeroes the bootstrap on terminal states, where there is no
future return to estimate.

Two details make this stable, and both are load bearing:

**The target network.** If $y$ were computed with $\theta$ itself, every gradient step would
move the target it is chasing. The regression problem would be nonstationary and the values
tend to diverge. Freezing $\theta^-$ for 10,000 updates turns each interval into ordinary
supervised regression against a fixed target.

**Experience replay.** Consecutive transitions in a game are heavily correlated, and SGD
assumes roughly i.i.d. samples. Transitions are written into a 100,000 entry buffer
(`torchrl` `LazyMemmapStorage`, backed by memory-mapped files so the buffer does not have to
fit in RAM) and minibatches are drawn uniformly at random. This decorrelates the updates and
lets each transition contribute to learning more than once.

### Exploration

Action selection is $\varepsilon$-greedy: with probability $\varepsilon$ pick uniformly at
random, otherwise take $\arg\max_a Q_\theta(s, a)$. After each episode,

$$
\varepsilon \leftarrow \max(\varepsilon_{\min}, \lambda \varepsilon)
$$

which is geometric decay toward a floor. The floor is not zero on purpose. A purely greedy
policy is deterministic, and a deterministic policy in a deterministic level can loop
forever against the same obstacle. Keeping $\varepsilon = 0.08$ preserves enough noise to
break out of those states, and evaluation uses the same value.

## Configuration

Hyperparameters live in `Model.__init__` in `model.py`.

| Parameter | Value | Notes |
| --- | --- | --- |
| Learning rate | 0.00025 | Adam |
| Discount $\gamma$ | 0.9 | Effective horizon $1/(1-\gamma) \approx 10$ decisions, or 40 game frames |
| $\varepsilon$ start / floor | 0.2 / 0.08 | Start is low because runs resume from trained weights |
| $\varepsilon$ decay | 0.9995 per episode | |
| Replay capacity | 100,000 | |
| Batch size | 32 | |
| Target sync interval | 10,000 gradient steps | |

A note on $\gamma = 0.9$: the agent effectively optimizes over the next 40 game frames, well
under a second. This is short for a platformer and biases the policy toward immediate
forward progress over setups that pay off later. Raising it to 0.95 or 0.99 lengthens the
horizon at the cost of slower and noisier convergence.

## Results

Reward history from the run that produced the current checkpoint (`episode_rewards.csv`,
36,878 episodes):

| Metric | Value |
| --- | --- |
| Mean reward, first 500 episodes | 646 |
| Mean reward, last 500 episodes | 1,735 |
| Best 100 episode moving average | 1,905 |
| Best single episode | 3,062 |

Reaching the flag is worth roughly 3,000, so the best episodes are full clears of 1-1 and
the trailing average sits a little past the halfway point of the level. Any episode where
`info["flag_get"]` is true is checkpointed separately as `model_ep_<n>_flag.pth`.

## Project layout

```
main.py                      Training loop, checkpoint resume, CSV logging
model.py                     Agent: replay buffer, epsilon-greedy, DQN update
ModelNN.py                   Convolutional Q-network
wrappers.py                  Frame skip, resize, grayscale, frame stack
trial_run_normal-speed.py    Evaluation at real time speed
curr-checkpoint.pth          Trained weights
episode_rewards.csv          episode, reward, epsilon per row
```

`checkpoints/` is gitignored. It is created on first run and holds every 500th episode,
plus every flag capture.

## Notes and limitations

- The update is single-estimator DQN. Because the same network both selects and evaluates
  the next action, $\max_{a'} Q_{\theta^-}(s', a')$ is biased upward. Double DQN fixes this
  by selecting with the online network and evaluating with the target one. That is a two
  line change in `Model.learn`.
- Only `SuperMarioBros-1-1` is trained. The policy is fit to one level and will not transfer.
- The action set is `RIGHT_ONLY` (5 actions). Backtracking is not possible, which is fine
  for 1-1 and a real limitation elsewhere.
- The evaluation script includes a stall breaker that forces a right press after 30 steps
  without x-position progress. It is a crutch for the deterministic-policy loop described
  above, not part of the learned behavior.

## License

MIT. See [LICENSE](LICENSE).
