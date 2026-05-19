# -*- coding: utf-8 -*-
#
# Usage:
#   python single_agent/train.py --agent dqn --episodes 2000
#   python single_agent/train.py --agent heuristic --episodes 200

#%% Import packages

import argparse
import sys
import time
from pathlib import Path

# Make project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import gymnasium as gym
from tqdm import tqdm

from shared_utils.paths import get_paths
from shared_utils.networks import DQNAgent
from shared_utils.logging_utils import (
    generate_run_id, stamped,
    save_single_hyperparameters, write_evaluation_log,
)
from single_agent.utils.plotting import plot_training_curve


#%% Functions

def train_single_dqn(env, agent, num_episodes, max_steps, print_every,
                     save_every, save_path):
    '''Train single DQN agent.'''
    rewards_history = []
    loss_history = []

    episodes = tqdm(range(num_episodes), desc='Training', unit='ep', dynamic_ncols=True)

    for episode in episodes:
        s, info = env.reset()
        total_reward = 0
        episode_losses = []
        steps = 0

        while steps < max_steps:
            a = agent.select_action(s, training=True)
            next_s, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            agent.store_transition(s, a, r, next_s, done)
            loss = agent.update()

            if loss is not None:
                episode_losses.append(loss)

            total_reward += r
            s = next_s
            steps += 1

            if done:
                break

        agent.decay_epsilon()
        agent.episodes_done += 1
        rewards_history.append(total_reward)

        if episode_losses:
            loss_history.append(np.mean(episode_losses))

        if episode % agent.target_update == 0:
            agent.update_target_network()

        if episode % save_every == 0 and episode > 0:
            agent.save(save_path)

    return rewards_history, loss_history


def run_dqn_training(episodes, max_steps, print_every, save_every,
                     load_model, run_post_eval):
    '''Train DQN agent end-to-end.'''
    paths = get_paths('single')
    run_id = generate_run_id()

    env = gym.make('LunarLander-v3')
    agent = DQNAgent()

    if load_model:
        agent.load(load_model)
        agent.epsilon = 1.0
        agent.epsilon_start = 1.0

    save_path = paths['models_dir'] / f'dqn_lunar_lander_{run_id}.pth'

    print()
    print('=' * 60)
    print('SINGLE-AGENT DQN TRAINING')
    print('=' * 60)
    print(f'Run ID:          {run_id}')
    print(f'Device:          {agent.device}')
    print(f'Episodes:        {episodes}')
    print(f'Max steps:       {max_steps}')
    print(f'Epsilon start:   {agent.epsilon_start}')
    print(f'Episodes done:   {agent.episodes_done}')
    print(f'Model:           {save_path}')
    print(f'Loaded from:     {load_model or "N/A"}')
    print('=' * 60)

    save_single_hyperparameters(
        agent,
        stamped(paths['logs_dir'], 'hyperparameters', '.txt', run_id),
    )

    t_start = time.time()
    rewards, losses = train_single_dqn(
        env, agent, episodes, max_steps,
        print_every, save_every, save_path,
    )
    elapsed = time.time() - t_start
    agent.save(save_path)
    env.close()

    plot_training_curve(
        rewards, losses,
        save_path=stamped(paths['plots_dir'], 'training', '.png', run_id),
    )

    h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
    print(f'\nTraining complete. Final avg reward: {np.mean(rewards[-100:]):.2f}')
    print(f'Training time: {h}h {m}m {s}s')

    if run_post_eval:
        from single_agent.run import run_single
        print('\nPost-Training Evaluation')
        run_single(agent_type='dqn', mode='evaluate', episodes=200,
                   model=str(save_path), run_id=run_id)
        print('\nPost-Training Demo')
        run_single(agent_type='dqn', mode='demo',
                   model=str(save_path), run_id=run_id)


def run_heuristic_training(episodes):
    print('Heuristic agent has no learnable parameters; dispatching to evaluation.')
    from single_agent.run import run_single
    run_single(agent_type='heuristic', mode='evaluate', episodes=episodes)


def run_training(agent_type, episodes=2000, max_steps=1000, print_every=10,
                 save_every=100, load_model=None, run_post_eval=True):
    '''Dispatch single-agent training based on agent type.'''
    if agent_type == 'dqn':
        run_dqn_training(episodes, max_steps, print_every, save_every,
                         load_model, run_post_eval)
    elif agent_type == 'heuristic':
        run_heuristic_training(episodes)
    else:
        raise ValueError(f'Unknown agent type {agent_type!r}. Choose dqn or heuristic.')


def main():
    parser = argparse.ArgumentParser(description='Train a single-agent Lunar Lander policy')
    parser.add_argument('--agent', required=True, choices=['dqn', 'heuristic'],
                        help='Agent type to train')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes (default: 2000)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max steps per episode (default: 1000)')
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to pre-trained agent weights (DQN only)')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip post-training evaluation and demo')

    args = parser.parse_args()
    run_training(
        agent_type=args.agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        print_every=args.print_every,
        save_every=args.save_every,
        load_model=args.load_model,
        run_post_eval=not args.no_eval,
    )


if __name__ == '__main__':

    if len(sys.argv) > 1:
        main()
    else:
        #%% IDE CONFIGURATION
        AGENT       = 'dqn'           # 'dqn' or 'heuristic'
        EPISODES    = 2000
        MAX_STEPS   = 1000
        PRINT_EVERY = 10
        SAVE_EVERY  = 100
        LOAD_MODEL  = None            # Path to agent weights, or None (DQN only)

        run_training(
            agent_type=AGENT,
            episodes=EPISODES,
            max_steps=MAX_STEPS,
            print_every=PRINT_EVERY,
            save_every=SAVE_EVERY,
            load_model=LOAD_MODEL,
        )

#%% End of Script