# src/algorithms/dqn_agent.py
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden=[512,256,128]):
        super(DQN, self).__init__()
        layers = []
        in_dim = state_size
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 batch_size=1024,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.997,
                 learning_rate=3e-4,
                 memory_size=100000,
                 device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.loss_history = []
        self.episode_rewards = []

    def remember(self, state, action, reward, next_state, done):
        # store raw floats/arrays (convert when sampling)
        self.memory.append((np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            bool(done)))

    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.model(state_tensor)
            return torch.argmax(qvals, dim=1).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([float(m[4]) for m in minibatch])).to(self.device)

        current_q = self.model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        # soft update
        self.soft_update_target_model(tau=0.005)
        # decay epsilon
        self.decay_epsilon()
        return loss.item()

    def soft_update_target_model(self, tau=0.005):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename, env=None):
        # Save safe state_dict + meta
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.state_size,
            'output_dim': self.action_size
        }
        if env is not None:
            save_data['action_space'] = getattr(env.action_space, 'n', None)
            save_data['observation_space'] = getattr(env.observation_space, 'shape', None)

        dirpath = os.path.dirname(filename)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        torch.save(save_data, filename)

    def load(self, filename, map_location=None):
        map_location = map_location or self.device
        ckpt = torch.load(filename, map_location=map_location)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.update_target_model()
        else:
            # support old plain state dict
            self.model.load_state_dict(ckpt)
            self.update_target_model()

    def plot_metrics(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        if self.loss_history:
            plt.figure()
            plt.plot(self.loss_history)
            plt.title("DQN training loss")
            plt.xlabel("Updates")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "dqn_loss.png"), dpi=200)
            plt.close()
        if self.episode_rewards:
            plt.figure()
            plt.plot(self.episode_rewards)
            plt.title("Episode rewards")
            plt.xlabel("Episode")
            plt.ylabel("Total reward")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "dqn_rewards.png"), dpi=200)
            plt.close()
