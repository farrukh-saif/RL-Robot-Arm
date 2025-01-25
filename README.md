# 🤖 AI-Powered Robotic Arm Control 🦾

A reinforcement learning project where AI agents learn to control a robotic arm in custom environments. Read the full story: **[Teaching Robots to Reach: A Reinforcement Learning Journey](https://syedfarrukhsaif.com/blog)**.

<img src="ai-playing.gif" width="400" alt="AI-controlled robotic arm reaching for targets">

## Overview

This project combines:
- 🏗️ Custom robotic arm environment built with **Gymnasium**
- 🧠 Deep reinforcement learning implemented via **Stable Baselines3**
- 📈 Training visualization with **TensorBoard**
- 🎮 Human-playable interface using **PyGame**

## Features

- 🏭 Custom 2D robotic arm environment with:
  - Realistic joint physics
  - Dynamic target generation
  - Multiple observation modes
- 🤖 PPO implementation for precise control
- 📊 Training progress visualization
- ✍️ **[Blog post](https://syedfarrukhsaif.com/blog)** documenting the entire journey

## Usage

**Manual control:**
```python
python robot_arm_env.py
```

**Train the agent:**
```python
python train.py
```

**Watch trained agent:**
```python
python use.py
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Farama Foundation for [Gymnasium](https://gymnasium.farama.org/)
- Stable Baselines3 team for their RL implementations
- All the coffee that powered this project ☕