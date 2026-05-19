# -*- coding: utf-8 -*-

# Usage:
#   python multi_agent/train.py --config partial --episodes 3000
#   python multi_agent/train.py --config full    --episodes 3000
#   python multi_agent/train.py --config mixed   --episodes 3000
#   python multi_agent/train.py --config partial --episodes 3000 --warm-start single_agent/Models/dqn_lunar_lander.pth

#%% Import packages

import argparse
import sys
import time
from pathlib import Path

# Make project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from shared_utils.paths import get_paths
from shared_utils.constants import COOP_SUCCESS_THRESHOLD
from shared_utils.logging_utils import (
    generate_run_id, stamped,
    save_coop_hyperparameters, write_evaluation_log,
)
from multi_agent.utils.agents import PartialObsAgent, FullObsAgent
from multi_agent.utils.coop_env import CooperativeLunarLander
from multi_agent.utils.plotting import plot_coop_training


#%% Variant configuration table

#!!! MUST mirror multi_agent/run.py
CONFIG_TABLE = {
    'partial': {
        'obs_mode': 'partial',
        'agent1_cls': PartialObsAgent,
        'agent2_cls': PartialObsAgent,
    },
    'full': {
        'obs_mode': 'full',
        'agent1_cls': FullObsAgent,
        'agent2_cls': FullObsAgent,
    },
    'mixed': {
        'obs_mode': 'mixed',
        'agent1_cls': PartialObsAgent,
        'agent2_cls': FullObsAgent,
    },
}


#%% Functions

def train_coop(coop_env, agent1, agent2, num_episodes, max_steps,
               print_every, save_every, save_path1, save_path2):
    '''Train two cooperative DQN agents simultaneously.'''
    rewards_history_1 = []
    rewards_history_2 = []
    loss_history_1 = []
    loss_history_2 = []
    joint_success_history = []

    episodes = tqdm(range(num_episodes), desc='Coop Training', unit='ep', dynamic_ncols=True)

    for episode in episodes:
        s1, s2 = coop_env.reset()
        total_r1, total_r2 = 0.0, 0.0
        ep_losses_1, ep_losses_2 = [], []

        for step in range(max_steps):
            a1 = agent1.select_action(s1, training=True)
            a2 = agent2.select_action(s2, training=True)

            (ns1, r1, done1), (ns2, r2, done2), both_done = coop_env.step(a1, a2)

            if not coop_env.done1 or done1:
                agent1.store_transition(s1, a1, r1, ns1, float(done1))
            if not coop_env.done2 or done2:
                agent2.store_transition(s2, a2, r2, ns2, float(done2))

            loss1 = agent1.update()
            loss2 = agent2.update()

            if loss1 is not None:
                ep_losses_1.append(loss1)
            if loss2 is not None:
                ep_losses_2.append(loss2)

            total_r1 += r1
            total_r2 += r2
            s1, s2 = ns1, ns2

            if both_done:
                break

        agent1.decay_epsilon()
        agent2.decay_epsilon()
        agent1.episodes_done += 1
        agent2.episodes_done += 1

        rewards_history_1.append(total_r1)
        rewards_history_2.append(total_r2)
        joint_success = (total_r1 > COOP_SUCCESS_THRESHOLD) and (total_r2 > COOP_SUCCESS_THRESHOLD)
        joint_success_history.append(joint_success)

        if ep_losses_1:
            loss_history_1.append(np.mean(ep_losses_1))
        if ep_losses_2:
            loss_history_2.append(np.mean(ep_losses_2))

        if episode % agent1.target_update == 0:
            agent1.update_target_network()
            agent2.update_target_network()

        if episode % save_every == 0 and episode > 0:
            agent1.save(save_path1)
            agent2.save(save_path2)

    return (rewards_history_1, rewards_history_2,
            loss_history_1, loss_history_2, joint_success_history)


def run_training(config, episodes=2000, max_steps=1000, print_every=10,
                 save_every=100, load_model_1=None, load_model_2=None,
                 warm_start=None, run_post_eval=True):
    '''Train two DQN agents for given variant.'''
    if config not in CONFIG_TABLE:
        raise ValueError(f'Unknown config {config!r}. Choose from {list(CONFIG_TABLE.keys())}.')

    cfg = CONFIG_TABLE[config]
    variant_key = f'coop_{config}'
    paths = get_paths(variant_key)
    run_id = generate_run_id()

    obs_mode = cfg['obs_mode']
    coop_env = CooperativeLunarLander(obs_mode=obs_mode)
    agent1 = cfg['agent1_cls']()
    agent2 = cfg['agent2_cls']()

    # Load pre-existing models or warm-start
    if load_model_1:
        agent1.load(load_model_1)
        agent1.epsilon = 1.0
        agent1.epsilon_start = 1.0
    elif warm_start:
        agent1.warm_start_from_single(warm_start)
    if load_model_2:
        agent2.load(load_model_2)
        agent2.epsilon = 1.0
        agent2.epsilon_start = 1.0
    elif warm_start:
        agent2.warm_start_from_single(warm_start)

    save_path1 = paths['models_dir'] / f'agent1_{agent1.state_size}d_{run_id}.pth'
    save_path2 = paths['models_dir'] / f'agent2_{agent2.state_size}d_{run_id}.pth'

    print()
    print('=' * 60)
    print('MULTI-AGENT COOPERATIVE TRAINING')
    print('=' * 60)
    print(f'Run ID:          {run_id}')
    print(f'Config:          {config}')
    print(f'Obs mode:        {obs_mode}')
    print(f'Device:          {agent1.device}')
    print(f'Episodes:        {episodes}')
    print(f'Max steps:       {max_steps}')
    print(f'Epsilon start:   {agent1.epsilon_start}')
    print(f'Episodes done:   {agent1.episodes_done}')
    print(f'Agent 1 model:   {save_path1}')
    print(f'Agent 2 model:   {save_path2}')
    print(f'Loaded from A1:  {load_model_1 or "N/A"}')
    print(f'Loaded from A2:  {load_model_2 or "N/A"}')
    print(f'Warm start:      {warm_start or "N/A"}')
    print('=' * 60)

    save_coop_hyperparameters(
        agent1, agent2, coop_env,
        stamped(paths['logs_dir'], 'hyperparameters', '.txt', run_id),
    )

    label1 = f'Agent 1 ({agent1.state_size}d)'
    label2 = f'Agent 2 ({agent2.state_size}d)'

    t_start = time.time()
    r1, r2, l1, l2, js = train_coop(
        coop_env, agent1, agent2, episodes, max_steps,
        print_every, save_every, save_path1, save_path2,
    )
    elapsed = time.time() - t_start
    agent1.save(save_path1)
    agent2.save(save_path2)
    coop_env.close()

    plot_coop_training(
        r1, r2, l1, l2, js,
        save_path=stamped(paths['plots_dir'], 'training', '.png', run_id),
        agent1_label=label1, agent2_label=label2,
    )

    write_evaluation_log(
        stamped(paths['logs_dir'], 'train', '.log', run_id),
        run_id, 'train', agent1, agent2, coop_env,
        rewards1=r1, rewards2=r2, joint_successes=js,
        model_path_1=load_model_1, model_path_2=load_model_2,
        warm_start_path=warm_start, num_episodes=episodes,
        variant=variant_key,
    )

    h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
    print('\nTraining complete.')
    print(f'Agent 1 final avg reward: {np.mean(r1[-100:]):.2f}')
    print(f'Agent 2 final avg reward: {np.mean(r2[-100:]):.2f}')
    print(f'Training time: {h}h {m}m {s}s')
    print(f'Joint success rate (last 100): {np.mean(js[-100:]) * 100:.1f}%')

    if run_post_eval:
        from multi_agent.run import run_multi
        print('\nPost-Training Evaluation')
        run_multi(config=config, mode='evaluate', episodes=200,
                  model_1=str(save_path1), model_2=str(save_path2),
                  run_id=run_id)
        print('\nPost-Training Demo')
        run_multi(config=config, mode='demo',
                  model_1=str(save_path1), model_2=str(save_path2),
                  run_id=run_id)


def main():
    parser = argparse.ArgumentParser(description='Train two cooperative DQN agents')
    parser.add_argument('--config', required=True,
                        choices=list(CONFIG_TABLE.keys()),
                        help='Knowledge configuration: partial, full, or mixed')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes (default: 2000)')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max steps per episode (default: 1000)')
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--load-model-1', type=str, default=None,
                        help='Path to pre-trained agent 1 weights')
    parser.add_argument('--load-model-2', type=str, default=None,
                        help='Path to pre-trained agent 2 weights')
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Single-agent weights for warm start')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip post-training evaluation and demo')

    args = parser.parse_args()
    run_training(
        config=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        print_every=args.print_every,
        save_every=args.save_every,
        load_model_1=args.load_model_1,
        load_model_2=args.load_model_2,
        warm_start=args.warm_start,
        run_post_eval=not args.no_eval,
    )


if __name__ == '__main__':

    if len(sys.argv) > 1:
        main()
    else:
        #%% IDE CONFIGURATION
        CONFIG        = 'full'         # 'partial', 'full', or 'mixed'
        EPISODES      = 3000
        MAX_STEPS     = 1000
        PRINT_EVERY   = 10
        SAVE_EVERY    = 100
        LOAD_MODEL_1  = None
        LOAD_MODEL_2  = None
        WARM_START    = 'single_agent/Models/dqn_lunar_lander.pth' # Example

        run_training(
            config=CONFIG,
            episodes=EPISODES,
            max_steps=MAX_STEPS,
            print_every=PRINT_EVERY,
            save_every=SAVE_EVERY,
            load_model_1=LOAD_MODEL_1,
            load_model_2=LOAD_MODEL_2,
            warm_start=WARM_START,
        )

#%% End of Script