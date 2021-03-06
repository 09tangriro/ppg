from collections import deque

import gym
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from helpers import exists
from ppg import PPG
from buffers import Memory


def main(
    env_name = 'LunarLander-v2',
    num_episodes = 5000,
    max_timesteps = 500,
    actor_hidden_dim = 32,
    critic_hidden_dim = 256,
    minibatch_size = 16,
    lr = 0.0005,
    betas = (0.9, 0.999),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 0.4,
    beta_s = .01,
    update_timesteps = 64,
    num_policy_updates_per_aux = 32,
    epochs = 1,
    epochs_aux = 6,
    render = False,
    render_every_eps = 250,
    save_every = 1000,
    load = True,
):
    """
    :param env_name: OpenAI gym environment name
    :param num_episodes: number of episodes to train
    :param max_timesteps: max timesteps per episode
    :param actor_hidden_dim: actor network hidden layer size
    :param critic_hidden_dim: critic network hidden layer size
    :param minibatch_size: minibatch size for training
    :param lr: learning rate for optimizers
    :param betas: betas for Adam Optimizer
    :param lam: GAE lambda (exponential discount)
    :param gamma: GAE gamma (future discount)
    :param eps_clip: PPO policy loss clip coefficient
    :param value clip: value loss clip coefficient
    :param beta_s: entropy loss coefficient
    :param update_timesteps: number of timesteps to run before training
    :param epochs: policy phase epochs
    :param epochs_aux: auxiliary phase epochs
    :param render: toggle render environment
    :param render_every_eps: if render, how often to render
    :param save_every: how often to save networks
    :load: toggle load a previously trained network 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    memories = deque([])
    aux_memories = deque([])

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

    if load:
        agent.load()

    time = 0
    updated = False
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc='episodes'):
        render_eps = render and eps % render_every_eps == 0
        state = env.reset()
        for timestep in range(max_timesteps):
            time += 1

            if updated and render_eps:
                env.render()

            state = torch.from_numpy(state).to(device)
            action_probs, _ = agent.actor(state)
            value = agent.critic(state)

            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            next_state, reward, done, _ = env.step(action)

            memory = Memory(state, action, action_log_prob, reward, done, value)
            memories.append(memory)

            state = next_state

            if time % update_timesteps == 0:
                agent.learn(memories, aux_memories, next_state)
                num_policy_updates += 1
                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

                updated = True

            if done:
                if render_eps:
                    updated = False
                break

        if render_eps:
            env.close()

        if eps % save_every == 0:
            agent.save()

if __name__ == "__main__":
    main()
