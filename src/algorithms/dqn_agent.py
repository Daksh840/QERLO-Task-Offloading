import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=512, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
                 learning_rate=5e-5, memory_size=20000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.loss_history = []
        self.episode_rewards = []

    def remember(self, state, action, reward, next_state, done):
        clipped_reward = np.clip(reward, -1.0, 1.0)
        self.memory.append((state, action, clipped_reward, next_state, done))

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in minibatch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch])).to(self.device)
        dones = torch.FloatTensor([float(e[4]) for e in minibatch]).to(self.device)

        current_q = self.model(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.soft_update_target_model(tau=0.005)
        self.decay_epsilon()

    def soft_update_target_model(self, tau=0.01):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename, env=None):
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.state_size,
            'output_dim': self.action_size
        }
        if env:
            save_data['action_space'] = env.action_space.n
            save_data['observation_space'] = env.observation_space.shape
        torch.save(save_data, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.update_target_model()

    def log_episode_reward(self, total_reward, episode):
        self.episode_rewards.append(total_reward)
        print(f"[Episode {episode}] Total Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.4f}")
