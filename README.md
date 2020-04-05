[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This solution is a DQN agent which can solve the banana collecting task.
It is a DDQN algorithm with a local neural network and a target neural network and a replay buffer
![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

With the provided implementation the environment can be solved in ~500 episodes, depending on the starting point. 
A pretrained agent is already provided in the checkpoint.pth which can be loaded in order to evaluate the pre-trained agent

### Getting Started

1. Clone the git repositroy into the p1_navigation Project provided by Udacity
2. Open the Navigation.ipynb in order to either train a new agent or see how the already trained agent behaves

### Instructions

Follow the instructions in `Navigation.ipynb` to either:

- Train a new agent or
- See how the pretrained agent behaves in the environment

The code cells which needs to be executed are documented in the Navigation.ipynb

