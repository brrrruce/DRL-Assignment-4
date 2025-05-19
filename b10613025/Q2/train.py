import argparse, time, os, random, math, itertools
from collections import deque, namedtuple

import numpy as np
import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from gymnasium import spaces
from shimmy import DmControlCompatibilityV0 as Dm2Gym

import torch
import torch.nn as nn
import torch.nn.functional as F

seed = None       
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class PixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, w=84, h=84):
        super().__init__(env)
        env.render_kwargs = {"width": w, "height": h, "camera_id": 0}
        tmp = env.render()
        self.observation_space = spaces.Box(0, 255, shape=tmp.shape, dtype=np.uint8)

    def observation(self, obs):       
        return self.env.render()

def make_env(flatten=True, use_pixels=False, seed=None):
    env = suite.load(
        "cartpole",
        "balance",
        task_kwargs={
            "random": seed if seed is not None else np.random.randint(1e6)
        },
    )
    env = Dm2Gym(
        env, render_mode="rgb_array",
        render_kwargs={"width": 256, "height": 256, "camera_id": 0}
    )
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)
    if use_pixels:
        env = PixelObservationWrapper(env)
    return env

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

    def sample(self, obs):
        dist = self._dist(obs)
        z = dist.rsample()
        u = torch.tanh(z)
        a = u * self.act_limit
        logp = dist.log_prob(z) - torch.log(1 - u.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True)

    def forward(self, obs):
        dist = self._dist(obs)
        return torch.tanh(dist.mean) * self.act_limit

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        def mlp():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1),
            )

        self.q1, self.q2 = mlp(), mlp()

    def forward(self, o, a):
        x = torch.cat([o, a], dim=-1)
        return self.q1(x), self.q2(x)

T = namedtuple("T", "o a r o2 d")

class Replay:
    def __init__(self, size):
        self.buf = deque(maxlen=size)

    def store(self, *args):
        self.buf.append(T(*args))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        o, a, r, o2, d = map(np.stack, zip(*batch))
        to = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
        return map(to, (o, a, r[:, None], o2, d[:, None]))

    def __len__(self):
        return len(self.buf)

class SAC:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        lr=3e-4,
        gamma=0.995,
        tau=0.005,
        auto_entropy=True,
    ):
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_t = Critic(obs_dim, act_dim).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma, self.tau = gamma, tau

        self.auto = auto_entropy
        if auto_entropy:
            self.target_entropy = -act_dim
            self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = 0.2

    @property
    def alpha(self):
        return self.log_alpha.exp() if self.auto else self._alpha

    @alpha.setter
    def alpha(self, v):
        self._alpha = torch.tensor(v, device=device)

    def act(self, o, deterministic=False):
        o = torch.as_tensor(o, device=device, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            with torch.no_grad():
                a = self.actor(o)
        else:
            with torch.no_grad():
                a, _ = self.actor.sample(o)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch):
        o, a, r, o2, d = batch


        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            q1t, q2t = self.critic_t(o2, a2)
            q_targ = torch.min(q1t, q2t) - self.alpha * logp2
            backup = r + self.gamma * (1 - d) * q_targ

        q1, q2 = self.critic(o, a)
        loss_q = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.q_opt.zero_grad()
        loss_q.backward()
        self.q_opt.step()

        a_pi, logp_pi = self.actor.sample(o)
        q1_pi, q2_pi = self.critic(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        self.pi_opt.zero_grad()
        loss_pi.backward()
        self.pi_opt.step()


        if self.auto:
            loss_alpha = -(self.log_alpha *
                           (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()


        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(),
                              self.critic_t.parameters()):
                p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

def evaluate(agent, episodes=100, seed=None):
    eval_seed = (seed + 2) if seed is not None else None
    env_eval = make_env(flatten=True, use_pixels=False, seed=eval_seed)

    returns = []
    for _ in range(episodes):
        o, _ = env_eval.reset(seed=None)
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(o, deterministic=True) 
            o, r, done, trunc, _ = env_eval.step(a)
            ep_ret += r
            done = done or trunc
        returns.append(ep_ret)
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    print(f"eval: {episodes} avg rew{mean_r:.2f} ± {std_r:.2f}")
    env_eval.close()
    return mean_r, std_r

def train(
    total_steps=100_000,
    batch=128,
    start_steps=10_000,
    eval_every=5_000,
    seed=None,
):
    env_seed = seed
    env_t = make_env(flatten=True, use_pixels=False, seed=env_seed)
    env_e = make_env(flatten=True, use_pixels=False, seed=(
        env_seed + 1) if env_seed is not None else None)

    obs_dim = env_t.observation_space.shape[0]
    act_dim = env_t.action_space.shape[0]
    act_lim = env_t.action_space.high[0]

    agent = SAC(obs_dim, act_dim, act_lim)
    replay = Replay(1_000_000)

    o, _ = env_t.reset(seed=env_seed)
    ep_ret, ep_len, returns = 0, 0, []
    best_score = -np.inf

    for t in range(1, total_steps + 1):
        a = env_t.action_space.sample() if t < start_steps else agent.act(o)
        o2, r, d, tr, _ = env_t.step(a)
        replay.store(o, a, r, o2, d or tr)
        o = o2
        ep_ret += r
        ep_len += 1

        if d or tr:
            o, _ = env_t.reset()
            returns.append(ep_ret)
            ep_ret = ep_len = 0


        if t >= start_steps and len(replay) >= batch:
            agent.update(replay.sample(batch))


        if t % eval_every == 0:
            rets = []
            for _ in range(100):
                oe, _ = env_e.reset()
                done = False
                rsum = 0
                while not done:
                    ae = agent.act(oe, deterministic=True)
                    oe, rn, done, tr, _ = env_e.step(ae)
                    rsum += rn
                    done = done or tr
                rets.append(rsum)

            avg = np.mean(rets)
            print(f"[{t:6d}] eval mean {avg:.2f} ± {np.std(rets):.2f}")
            if avg > 985 and avg > best_score:
                best_score = avg
                torch.save(agent.actor.state_dict(),
                           f"cartpole_actor.pth")
                torch.save(agent.critic.state_dict(),
                           f"cartpole_critic.pth")


    env_t.close()
    env_e.close()

if __name__ == "__main__":
    train(seed=seed)
