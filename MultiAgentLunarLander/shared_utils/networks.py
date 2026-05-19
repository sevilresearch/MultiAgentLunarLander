# -*- coding: utf-8 -*-
# shared_utils/networks.py
# DQNNetwork, ReplayBuffer, DQNAgent

#%% Import packages

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


#%% Class definitions

class DQNNetwork(nn.Module):
    '''Simple feedforward network for Q-value approximation.'''

    def __init__(self, state_size=8, action_size=4, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    '''Experience replay buffer.'''

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    '''DQN Agent for Lunar Lander.'''

    def __init__(self,
                 state_size=8,
                 action_size=4,
                 hidden_size=64,
                 lr=0.00063,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.07,
                 epsilon_decay=0.999,
                 buffer_size=100000,
                 batch_size=64,
                 target_update=20,
                 device=None
                 ):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Networks
        self.policy_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(self, state, training=True):
        '''Epsilon-greedy action selection.'''
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        '''Store transition in replay buffer.'''
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        '''Perform one step of optimization.'''
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q(s, a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute max Q(s', a') for next states
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss and update
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        '''Decay exploration rate.'''
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        '''Copy weights from policy net to target net.'''
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        '''Save model weights.'''
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'episodes_done': self.episodes_done
        }, filepath)

    def load(self, filepath):
        '''Load model weights.'''
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon)
        self.episodes_done = checkpoint['episodes_done']

#%% End of Script