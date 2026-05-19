# -*- coding: utf-8 -*-
# shared_utils/logging_utils.py
# Support module containing utility and logging functions

#%% Import packages

from datetime import datetime
from pathlib import Path

import numpy as np

from shared_utils.constants import (
    COOP_SUCCESS_THRESHOLD, LANDER1_SPAWN_X_OFFSET, LANDER2_SPAWN_X_OFFSET,
)


#%% Functions

def generate_run_id():
    '''Generate a timestamp-based run ID (e.g., 2026-03-21_143052).'''
    return datetime.now().strftime('%Y-%m-%d_%H%M%S')


def stamped(directory, basename, ext, run_id):
    '''Build a timestamped filepath: directory / basename_runid.ext'''
    return Path(directory) / f'{basename}_{run_id}{ext}'


def save_coop_hyperparameters(agent1, agent2, coop_env, save_path):
    '''Save cooperative hyperparameters to file.'''
    with open(save_path, 'w') as f:
        f.write('COOPERATIVE DQN HYPERPARAMETERS\n')
        f.write('=' * 40 + '\n')
        f.write(f'state_size: {agent1.state_size}\n')
        f.write(f'action_size: {agent1.action_size}\n')
        f.write(f'hidden_size: {agent1.policy_net.fc1.out_features}\n')
        f.write(f'lr: {agent1.optimizer.param_groups[0]["lr"]}\n')
        f.write(f'gamma: {agent1.gamma}\n')
        f.write(f'epsilon_start: {agent1.epsilon_start}\n')
        f.write(f'epsilon_end: {agent1.epsilon_end}\n')
        f.write(f'epsilon_decay: {agent1.epsilon_decay}\n')
        f.write(f'buffer_size: {agent1.buffer.buffer.maxlen}\n')
        f.write(f'batch_size: {agent1.batch_size}\n')
        f.write(f'target_update: {agent1.target_update}\n')
        f.write(f'device: {agent1.device}\n')

        if agent2.state_size != agent1.state_size:
            f.write(f'\nAgent 2 state_size: {agent2.state_size}\n')

        f.write('\nCOOPERATIVE PARAMS\n')
        f.write('=' * 40 + '\n')
        if coop_env is not None:
            f.write(f'obs_mode: {coop_env.obs_mode}\n')
            f.write(f'zone_penalty_scale: {coop_env.zone_penalty_scale}\n')
            f.write(f'zone_bonus: {coop_env.zone_bonus}\n')
            f.write(f'contact_penalty: {coop_env.contact_penalty}\n')
            f.write(f'hover_window_size: {coop_env.hover_window_size}\n')
            f.write(f'hover_progress_thresh_x: {coop_env.hover_progress_thresh_x}\n')
            f.write(f'hover_progress_thresh_y: {coop_env.hover_progress_thresh_y}\n')
            f.write(f'hover_base_penalty: {coop_env.hover_base_penalty}\n')
            f.write(f'hover_penalty_exponent: {coop_env.hover_penalty_exponent}\n')
            f.write(f'hover_max_penalty: {coop_env.hover_max_penalty}\n')
            f.write(f'hover_min_altitude: {coop_env.hover_min_altitude}\n')
            f.write(f'hover_counter_decay: {coop_env.hover_counter_decay}\n')
        f.write(f'lander1_spawn_offset: {LANDER1_SPAWN_X_OFFSET}\n')
        f.write(f'lander2_spawn_offset: {LANDER2_SPAWN_X_OFFSET}\n')
        f.write(f'coop_success_threshold: {COOP_SUCCESS_THRESHOLD}\n')


def save_single_hyperparameters(agent, save_path):
    '''Save single-agent hyperparameters to file.'''
    with open(save_path, 'w') as f:
        f.write('DQN HYPERPARAMETERS\n')
        f.write(f'hidden_size: {agent.policy_net.fc1.out_features}\n')
        f.write(f'lr: {agent.optimizer.param_groups[0]["lr"]}\n')
        f.write(f'gamma: {agent.gamma}\n')
        f.write(f'epsilon_start: {agent.epsilon_start}\n')
        f.write(f'epsilon_end: {agent.epsilon_end}\n')
        f.write(f'epsilon_decay: {agent.epsilon_decay}\n')
        f.write(f'buffer_size: {agent.buffer.buffer.maxlen}\n')
        f.write(f'batch_size: {agent.batch_size}\n')
        f.write(f'target_update: {agent.target_update}\n')
        f.write(f'device: {agent.device}\n')


def write_evaluation_log(log_path, run_id, mode, agent1, agent2, coop_env,
                         rewards1=None, rewards2=None, joint_successes=None,
                         model_path_1=None, model_path_2=None,
                         warm_start_path=None, num_episodes=None,
                         variant=None):
    '''Write a detailed evaluation log for the run.

    Handles both single-agent (agent2=None, coop_env=None, rewards2=None)
    and cooperative (all populated) cases.
    '''
    with open(log_path, 'w') as f:
        f.write(f'LUNAR LANDER - {mode.upper()} LOG\n')
        f.write('=' * 60 + '\n')
        f.write(f'Run ID:    {run_id}\n')
        f.write(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Mode:      {mode}\n')
        if variant:
            f.write(f'Variant:   {variant}\n')
        f.write('\n')

        # Agent model info
        f.write('MODEL FILES\n')
        f.write('-' * 40 + '\n')
        f.write(f'Agent 1 weights: {model_path_1 or "N/A"}\n')
        f.write(f'Agent 2 weights: {model_path_2 or "N/A"}\n')
        f.write(f'Warm start from: {warm_start_path or "N/A"}\n')
        f.write('\n')

        # Network architecture
        f.write('NETWORK ARCHITECTURE\n')
        f.write('-' * 40 + '\n')
        if agent1 is not None:
            f.write(f'Agent 1 state size: {agent1.state_size}\n')
            if agent2 is not None:
                f.write(f'Agent 2 state size: {agent2.state_size}\n')
            f.write(f'Action size:   {agent1.action_size}\n')
            f.write(f'Hidden size:   {agent1.policy_net.fc1.out_features}\n')
            f.write(f'Device:        {agent1.device}\n')
        else:
            f.write('(no learnable network -- non-DQN policy)\n')
        f.write('\n')

        # Hyperparameters
        f.write('HYPERPARAMETERS\n')
        f.write('-' * 40 + '\n')
        if agent1 is not None:
            f.write(f'Learning rate:  {agent1.optimizer.param_groups[0]["lr"]}\n')
            f.write(f'Gamma:          {agent1.gamma}\n')
            f.write(f'Epsilon:        {agent1.epsilon_start}\n')
            f.write(f'Epsilon end:    {agent1.epsilon_end}\n')
            f.write(f'Epsilon decay:  {agent1.epsilon_decay}\n')
            f.write(f'Buffer size:    {agent1.buffer.buffer.maxlen}\n')
            f.write(f'Batch size:     {agent1.batch_size}\n')
            f.write(f'Target update:  {agent1.target_update}\n')
            f.write(f'Episodes done (agent1): {agent1.episodes_done}\n')
            if agent2 is not None:
                f.write(f'Episodes done (agent2): {agent2.episodes_done}\n')
        else:
            f.write('(no learnable parameters)\n')
        f.write('\n')

        # Environment configuration
        f.write('ENVIRONMENT CONFIG\n')
        f.write('-' * 40 + '\n')
        if coop_env is not None:
            f.write(f'Obs mode:               {coop_env.obs_mode}\n')
            f.write(f'Zone penalty scale:     {coop_env.zone_penalty_scale}\n')
            f.write(f'Zone bonus:             {coop_env.zone_bonus}\n')
            f.write(f'Contact penalty:        {coop_env.contact_penalty}\n')
            f.write(f'Hover window size:      {coop_env.hover_window_size}\n')
            f.write(f'Hover thresh X:         {coop_env.hover_progress_thresh_x}\n')
            f.write(f'Hover thresh Y:         {coop_env.hover_progress_thresh_y}\n')
            f.write(f'Hover base penalty:     {coop_env.hover_base_penalty}\n')
            f.write(f'Hover penalty exponent: {coop_env.hover_penalty_exponent}\n')
            f.write(f'Hover max penalty:      {coop_env.hover_max_penalty}\n')
            f.write(f'Hover min altitude:     {coop_env.hover_min_altitude}\n')
            f.write(f'Hover counter decay:    {coop_env.hover_counter_decay}\n')
            f.write(f'Lander 1 spawn offset:  {LANDER1_SPAWN_X_OFFSET}\n')
            f.write(f'Lander 2 spawn offset:  {LANDER2_SPAWN_X_OFFSET}\n')
            f.write(f'Success threshold:      {COOP_SUCCESS_THRESHOLD}\n')
        else:
            f.write('Env:               gymnasium LunarLander-v3 (single)\n')
            f.write('Success threshold: 200\n')
        f.write('\n')

        # Results - if available. Cooperative threshold is COOP_SUCCESS_THRESHOLD
        # Single-agent uses Gymnasium's solved threshold of 200.
        if rewards1 is not None:
            threshold = COOP_SUCCESS_THRESHOLD if rewards2 is not None else 200
            f.write('RESULTS\n')
            f.write('-' * 40 + '\n')
            f.write(f'Episodes evaluated: {num_episodes or len(rewards1)}\n')
            f.write('\n')
            f.write('Agent 1:\n')
            f.write(f'  Mean reward:  {np.mean(rewards1):+.2f}\n')
            f.write(f'  Std reward:   {np.std(rewards1):.2f}\n')
            f.write(f'  Min reward:   {np.min(rewards1):+.2f}\n')
            f.write(f'  Max reward:   {np.max(rewards1):+.2f}\n')
            a1_solved = sum(r > threshold for r in rewards1)
            f.write(f'  Solved (>{threshold}): {a1_solved}/{len(rewards1)} '
                    f'({a1_solved / len(rewards1) * 100:.1f}%)\n')
            f.write('\n')
            if rewards2 is not None:
                f.write('Agent 2:\n')
                f.write(f'  Mean reward:  {np.mean(rewards2):+.2f}\n')
                f.write(f'  Std reward:   {np.std(rewards2):.2f}\n')
                f.write(f'  Min reward:   {np.min(rewards2):+.2f}\n')
                f.write(f'  Max reward:   {np.max(rewards2):+.2f}\n')
                a2_solved = sum(r > threshold for r in rewards2)
                f.write(f'  Solved (>{threshold}): {a2_solved}/{len(rewards2)} '
                        f'({a2_solved / len(rewards2) * 100:.1f}%)\n')
                f.write('\n')

                if joint_successes is not None:
                    joint_pct = sum(joint_successes) / len(joint_successes) * 100
                    f.write(f'Joint success:  {sum(joint_successes)}/{len(joint_successes)} '
                            f'({joint_pct:.1f}%)\n')

        f.write('\n' + '=' * 60 + '\n')
        f.write('END OF LOG\n')

    return log_path


#%% End of Script
