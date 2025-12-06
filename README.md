# Mario DDQN Trainer & Eval

A simple Deep Q-Learning setup for Super Mario Bros (gym_super_mario_bros + nes-py) with training (`main.py`) and evaluation (`eval_zero.py`). Includes wrappers for frame processing and checkpointing so you can pause/resume runs and track reward history.

## Latest Checkpoint
- checkpoints/model_ep_9961_flag.pth

## Python Version
- Tested with Python 3.10–3.11 (recommended 3.11).

## Setup
1) Create/activate a virtual environment (optional but recommended).
2) Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3) Verify ROM/environment runs:
   ```bash
   python -c "import gym_super_mario_bros; print('gym OK')"
   ```

## Training
- Run: `python main.py`
- Resumes from the latest checkpoint automatically.
- Logs rewards to `episode_rewards.csv` (includes epsilon per episode).

### Epsilon settings (exploration)
- Training: see `model.py` (`epsilon_start`, `eps_decay`, `eps_min`). Tweaking these changes how quickly the agent exploits vs explores.
- Eval: `eval_zero.py` sets a small epsilon (`model.epsilon = 0.05`) to avoid getting stuck; set to `0.0` for pure greedy play, or higher to add randomness.

## Evaluation
- Run: `python eval_zero.py`
- Uses the same wrappers as training to keep behavior consistent.
- Slight frame pacing via `target_frame_time` for playback speed.

## Reward Log (CSV)
- File: `episode_rewards.csv`
- To reset it, delete the file (or empty it) before running; kept here for transparency/history.

## Notes / Next Steps
- The current policy still needs more training; expect further improvement with additional episodes.
- Checkpoints save every 500 episodes and whenever the flag is reached; feel free to prune older checkpoints if disk space is an issue.
