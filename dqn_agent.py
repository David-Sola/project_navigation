import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# Global parameters for the classes
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 100  # minibatch size
UPDATE_EVERY = 10  # how often to update the network

class Agent():
    def __init__(self, state_size, action_size, alpha, gamma, tau):
        ''' Create a class called agent, which is the main part for our Depp - learning task
            We have subfunctionalities to:
            - Act in the specified environment
                --> def act(self, state, epsilon):
            - Get the feedback from the environment and save the state, action reward, next state tuples
            inside a memoris
                --> def step(self, state, action, reward, next_state):
            - A learning step where we take a random sampled batch from our replay memory calculating the TD - error
            and doinng the optimization step in order to adjust the weights of the neural network
                --> def learn(self, experiences, gamma):
        Params:
        =======
            state_size: Size of the states provided by the environment
            action_size: Size of the possible actions for the environment
            alpha: Learning rate
            gamma: Discount rate for future rewards
            tau: Value to slowly adjust the weights of the target network to the local network
        '''
        
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        # Creation of a neural network specified in model.py
        self.model_local = QNetwork(state_size, action_size)
        self.model_target = QNetwork(state_size, action_size)
        # Specifying the optimizer for the local network
        self.optimizer = optim.Adam(self.model_local.parameters(), self.alpha)
        # Initializing the replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Variable to later determine, when to do a learning step
        self.t_step = 0
        
    def step(self, state, action, reward, next_state):
        '''Get´s the information about the state of the environment, and stores it in the memory.
        Additionaly the function determines, if a learning steps needs to be taken or not

        Params:
        ======
            state: Current State of the environment
            action: The action taken in the current state
            reward: The reward that has been granted by taking the action in the current state
            next_state: The next state reached by taking the action in the previous state'''

        # Adding the state, action, reward, next_state pair into te replay memory
        self.memory.add(state, action, reward, next_state)
        # Check if "UPDATE_EVERY" (at the moment 10) steps have been taken. If yes continue by getting a sample
        # out of the replay memory and performing a learning step
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Check if enough samples are inside the the memory to get out a batch subset to learn from it
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
                
    def learn(self, experiences, gamma):
        '''
        The learning function of the agent. Here the TD - error fort the complete batch will be calculated. Afterwards
        the mean squared error will be calculated to do a backpropagation through the neural network.
        At the end the weights of the target network will be adjusted
        :param experiences: a sample of experiences from the replay memory with the size BATCH_SIZE
        :param gamma: Discount rate for future rewards
        '''

        # Extrace the states, actions, rewards and next_states from the experience memory
        states, actions, rewards, next_states = experiences
        # Calculate the TD - target
        Q_target_next = self.model_target(next_states).max(1)[0].unsqueeze(1)
        Q_target = rewards + (gamma * Q_target_next)
        # Changing the actions from float to integer
        actions = actions.long()
        # Calculating the old Q-values
        Q_expected = self.model_local(states).gather(1, actions)
        # Calculating the means squared error loss
        loss = F.mse_loss(Q_expected, Q_target)
        # First setting the rgadients to zero, as otherwise they´re being accumulated, then doing the backpropagation
        # and an optimize step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update target network
        for target_param, local_param in zip(self.model_target.parameters(), self.model_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)       
        
        
    def act(self, state, epsilon):
        '''
        Functionality to determine the action in the current state. As epsilon decays over time, we´re starting with
        lot´s of random actions that get more less over time and the estimated action of the network will be taken
        :param state: Current state of the environment
        :param epsilon: Value to slowly change from random sampling to greedy polica
        :return: The action to take in the environment
        '''
        # Get the action inside the current state. Choose from time to time a random action.
        # The chance to use a random action gets smaller when epsilon gets smaller
        if random.random() > epsilon:
            # Change the state to a tensor
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            # Eval the neural network for the current state but without adjusting the weights
            self.model_local.eval()
            with torch.no_grad():
                action_value = self.model_local(state_t)
            self.model_local.train()
            # Convert the action from tensor to numpy value
            return np.argmax(action_value.data.numpy())
        # Take a random action
        else:
            return np.random.randint(self.action_size)
            
        
        
class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size):
        '''
        Class for the replay buffer. Inside the replay buffer memories of interacting with the environment will be
        stored. A memory is always a state, action, reward, next state tuple. For the learning step a random sample
        will be extracted from the replay buffer
        :param action_size: Size of the possible actions for the environment
        :param buffer_size: Size of the whole replay buffer
        :param batch_size: Size of the batch for training
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        
    def add(self, state, action, reward, next_state):
        '''
        Adding a new state, action, reward, next_state tuple to the replay memory
        :param state: Current state
        :param action: Action taken in current stat
        :param reward: Reward that has been granted
        :param next_state: Next state reached
        :return:
        '''
        self.memory.append(self.experience(state, action, reward, next_state))
    
    def sample(self):
        '''
        Extracting a set of states, actions, rewards, next states by the size of the batch size from the replay buffer
        :return: Return the states, actions, rewards, next_states (size = batch_size)
        '''
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
    
        return (states, actions, rewards, next_states)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        