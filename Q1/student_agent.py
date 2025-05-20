import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# Do not modify the input of the 'act' function and the '__init__' function. 
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        mu = self.mu_layer(x)
        return torch.tanh(mu) * self.act_limit

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.device = torch.device("cpu")

        self.actor: Actor | None = None
        self._initialized: bool = False
    def _lazy_init(self, obs_dim: int) -> None:

        act_dim = self.action_space.shape[0]
        act_limit = float(self.action_space.high[0])


        self.actor = Actor(obs_dim, act_dim, act_limit).to(self.device)
        try:
            state_dict = torch.load("actor.pth", map_location=self.device)
            self.actor.load_state_dict(state_dict)
            self.actor.eval()
            self._initialized = True
        except FileNotFoundError:
            self.actor = None
            self._initialized = False

    def act(self, observation):
        action = self.action_space.sample()
        return action.astype(np.float32)
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


        return action.astype(np.float32)
        