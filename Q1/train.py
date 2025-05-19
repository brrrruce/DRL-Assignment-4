import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)
        self.act_limit = act_limit

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        std = log_std.exp().clamp(min=1e-6)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        u = torch.tanh(z)               
        action = u * self.act_limit             
        logp = dist.log_prob(z) - torch.log(1 - u.pow(2) + 1e-6)
        logp = logp.sum(dim=-1, keepdim=True)
        return action, logp

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        xu = torch.cat([obs, act], dim=-1)
        return self.q1(xu), self.q2(xu)


Transition = namedtuple('Transition',
                        ['obs','act','rew','obs2','done'])
class ReplayBuffer:
    def __init__(self, size):
        self.buf = deque(maxlen=size)
    def store(self, *args):
        self.buf.append(Transition(*args))
    def sample_batch(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buf)


class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit):
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.pi_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.q_optimizer  = torch.optim.Adam(self.critic.parameters(),  lr=1e-3)
        self.alpha = 0.1
        self.gamma = 0.98
        self.tau   = 0.01
        self.act_limit = act_limit

    def update(self, batch):
        obs = torch.tensor(batch.obs,  dtype=torch.float32, device=device)
        act = torch.tensor(batch.act,  dtype=torch.float32, device=device)
        rew = torch.tensor(batch.rew,  dtype=torch.float32, device=device).unsqueeze(-1)
        obs2= torch.tensor(batch.obs2, dtype=torch.float32, device=device)
        done= torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(-1)

        with torch.no_grad():
            a2, logp2 = self.actor.sample(obs2)
            q1_t, q2_t = self.critic_target(obs2, a2)
            q_targ = torch.min(q1_t, q2_t) - self.alpha * logp2
            backup = rew + self.gamma * (1 - done) * q_targ
        q1, q2 = self.critic(obs, act)

        loss_q = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.q_optimizer.step()


        a, logp = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, a)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.alpha * logp - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  max_norm=1.0)
        self.pi_optimizer.step()

        for p, p_targ in zip(self.critic.parameters(),
                              self.critic_target.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mu, std = self.actor.forward(obs)
        if deterministic:
            a = torch.tanh(mu) * self.act_limit
        else:
            a, _ = self.actor.sample(obs)
        return a.detach().cpu().numpy()[0]


def train():
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = SACAgent(obs_dim, act_dim, act_limit)
    buf = ReplayBuffer(size=500_000)

    total_steps = 200_000
    start_steps = 10_000
    batch_size = 128
    update_after = start_steps
    update_every = 1

    obs, _ = env.reset()
    for t in range(total_steps):
        if t < start_steps:
            act = env.action_space.sample()
        else:
            act = agent.select_action(obs)

        obs2, rew, done, truncated, _ = env.step(act)
        buf.store(obs, act, rew, obs2, done or truncated)
        obs = obs2

        if done or truncated:
            obs, _ = env.reset()

        if t >= update_after and t % update_every == 0 and len(buf) >= batch_size:
            batch = buf.sample_batch(batch_size)
            agent.update(batch)

    torch.save(agent.actor.state_dict(), "actor.pth")
    torch.save(agent.critic.state_dict(), "critic.pth")
    env.close()

if __name__ == "__main__":
    start = time.time()
    train()
    
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    agent = SACAgent(obs_dim, act_dim, act_limit)

    agent.actor.load_state_dict(torch.load("actor.pth"))
    agent.critic.load_state_dict(torch.load("critic.pth"))


    rewards = []
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            #action = env.action_space.sample()    #random
            obs, rew, done, truncated, _ = env.step(action)
            ep_ret += rew
            done = done or truncated
        rewards.append(ep_ret)
    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    print(f"avg rew = {mean_r:.2f} ± {std_r:.2f}")

    from google.colab import files
    #files.download("actor.pth")
    #files.download("critic.pth")
    env.close()