import gym
import torch
import numpy as np
from torch.distributions import Categorical

from ppg import PPG


def test_agent(env_name = 'LunarLander-v2', 
               num_timesteps=10000,
               actor_hidden_dim = 32,
               critic_hidden_dim = 256,
               minibatch_size = 16,
               epochs = 1,
               epochs_aux = 6,
               lr = 0.0005,
               betas = (0.9, 0.999),
               lam = 0.95,
               gamma = 0.99,
               eps_clip = 0.2,
               value_clip = 0.4,
               beta_s = .01,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = PPG(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        device
    )

    agent.load()

    state = env.reset()
    for _ in range(num_timesteps):
        state = torch.Tensor(state[np.newaxis, :]).to(device)
        action_probs, _ = agent.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample().item()
        state, rewards, dones, info = env.step(action)
        env.render()
        if dones == True:
            env.reset()
    env.close()

test_agent()