import gymnasium
import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.act_limit = act_limit

    def _dist(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std)

    def forward(self, obs):
        dist = self._dist(obs)
        return torch.tanh(dist.mean) * self.act_limit

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.device = torch.device("cpu")
        self.actor = None
        self._initialized = False

    def _lazy_init(self, obs_dim):
        act_dim = self.action_space.shape[0]
        act_limit = float(self.action_space.high[0])
        self.actor = Actor(obs_dim, act_dim, act_limit).to(self.device)
        try:
            state_dict = torch.load("cartpole_actor.pth", map_location=self.device)
            self.actor.load_state_dict(state_dict)
            self.actor.eval()
            self._initialized = True
        except Exception:
            self.actor = None
            self._initialized = False

    def act(self, observation):
        if not isinstance(observation, np.ndarray):
            observation = np.asarray(observation, dtype=np.float32)

        if self.actor is None:
            self._lazy_init(obs_dim=observation.shape[0])

        if self._initialized and self.actor is not None:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
                action = self.actor(obs_tensor)[0].cpu().numpy()
        else:
            action = self.action_space.sample()

        return action.astype(np.float64)
