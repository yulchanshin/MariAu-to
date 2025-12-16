<<<<<<< HEAD
# 🍄 MariAu-to: Super Mario Bros DQN Agent

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**MariAu-to** is a Deep Q-Network (DQN) agent capable of learning to play *Super Mario Bros* using raw pixel inputs. Built with PyTorch and `gym-super-mario-bros`, this project demonstrates reinforcement learning concepts including experience replay, target networks, and curriculum learning.

## ✨ Features
=======
# Mario DDQN Trainer & Eval

A Double Deep Q-Learning setup for Super Mario Bros (gym_super_mario_bros + nes-py) with training (`main.py`) and evaluation (`eval_zero.py`). Includes wrappers for frame processing and checkpointing so you can pause/resume runs and track reward history.

## Latest Checkpoint
- [checkpoints/model_ep_36890_flag.pth](https://github.com/yulchanshin/MariAu-to/blob/main/curr-checkpoint.pth)

## Python Version
- Tested with Python 3.10–3.11 (recommended 3.11).
>>>>>>> a4ffbb1013a08e4f6916cddb2ff5c705f041d717

- **Double Deep Q-Learning (DDQN)** using a CNN-based policy network.
- **Custom Environment Wrappers** for frame skipping, resizing (84x84), and grayscale processing.
- **Checkpointing System** that automatically resumes from the latest episode.
- **Performance Logging** tracking episode rewards and epsilon decay in `episode_rewards.csv`.
- **Pre-trained Flag Detection**: Saves special checkpoints when the agent reaches the flag.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MariAu-to.git
   cd MariAu-to
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

<<<<<<< HEAD
## 🚀 Usage
=======
## Training
- Run: `python main.py`
- Resumes from the latest checkpoint automatically. (If it's the first time, it will automatically create a folder)
- Logs rewards to `episode_rewards.csv` (includes epsilon per episode)
>>>>>>> a4ffbb1013a08e4f6916cddb2ff5c705f041d717

### Training the Agent
To start training the agent from scratch or resume from the latest checkpoint, simply run:

```bash
python main.py
```
The script will:
- Load the latest checkpoint from `checkpoints/` (if available).
- Log rewards to `episode_rewards.csv`.
- Render the gameplay in a window (set `render_mode='human'` in `main.py` if you want to watch while training).

### Watching the Agent Play
To watch a trained agent play continuously without training:

```bash
python trial_run_normal-speed.py
```

## 📂 Project Structure

- `main.py`: Entry point for training the DQN agent. Handles the training loop, epsilon decay, and logging.
- `model.py`: Defines the `Model` class (CNN architecture) and memory replay buffer.
- `wrappers.py`: Custom Gym wrappers to process observations (grayscale, resize, stack frames).
- `checkpoints/`: Directory where `.pth` model weights are saved automatically.
- `episode_rewards.csv`: Log file containing reward history for analysis.

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
