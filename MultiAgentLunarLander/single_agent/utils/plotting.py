# -*- coding: utf-8 -*-
# single_agent/utils/plotting.py
# Single DQN agent training and evaluation plots

#%% Import packages

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Consistent y-axis limits across all reward plots
REWARD_YLIM = (-500, 500)

def _dynamic_ylim(rewards, padding=0.1):
    '''Compute y-axis limits with padding from actual data range.'''
    rmin, rmax = np.min(rewards), np.max(rewards)
    span = rmax - rmin if rmax > rmin else 50
    return (rmin - padding * span, rmax + padding * span)


#%% Functions

def plot_training_curve(training_rewards, loss_history=None, window=50,
                        save_path=None, reward_ylim=None,
                        agent_label='DQN'):
    '''Plot DQN training curve (rewards + optional loss).'''

    if reward_ylim is None:
        reward_ylim = _dynamic_ylim(training_rewards)

    episodes = range(1, len(training_rewards) + 1)
    ncols = 2 if loss_history else 1

    # Combined subplot figure
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Rewards subplot
    ax = axes[0]
    ax.plot(episodes, training_rewards, alpha=0.3, color='blue', label='Episode reward')
    if len(training_rewards) >= window:
        moving_avg = np.convolve(training_rewards, np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(training_rewards) + 1), moving_avg,
                color='red', linewidth=2, label=f'{window}-episode moving avg')
    ax.axhline(y=200, color='green', linestyle='--', label='Solved threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'{agent_label} Training Rewards')
    ax.set_ylim(reward_ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Loss subplot
    if loss_history:
        ax = axes[1]
        loss_eps = range(1, len(loss_history) + 1)
        ax.plot(loss_eps, loss_history, alpha=0.3, color='orange', label='Episode loss')
        if len(loss_history) >= window:
            moving_avg = np.convolve(loss_history, np.ones(window) / window, mode='valid')
            ax.plot(range(window, len(loss_history) + 1), moving_avg,
                    color='red', linewidth=2, label=f'{window}-episode moving avg')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title(f'{agent_label} Training Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    # Individual plots
    if save_path:
        sp = Path(save_path)

        # Training rewards
        fig_r, ax_r = plt.subplots(figsize=(10, 6))
        ax_r.plot(episodes, training_rewards, alpha=0.3, color='blue', label='Episode reward')
        if len(training_rewards) >= window:
            moving_avg = np.convolve(training_rewards, np.ones(window) / window, mode='valid')
            ax_r.plot(range(window, len(training_rewards) + 1), moving_avg,
                      color='red', linewidth=2, label=f'{window}-episode moving avg')
        ax_r.axhline(y=200, color='green', linestyle='--', label='Solved threshold')
        ax_r.set_xlabel('Episode')
        ax_r.set_ylabel('Reward')
        ax_r.set_title(f'{agent_label} Training Rewards')
        ax_r.set_ylim(reward_ylim)
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_rewards')
        plt.savefig(p, dpi=150)
        plt.close(fig_r)

        # Training loss
        if loss_history:
            fig_l, ax_l = plt.subplots(figsize=(10, 6))
            loss_eps = range(1, len(loss_history) + 1)
            ax_l.plot(loss_eps, loss_history, alpha=0.3, color='orange', label='Episode loss')
            if len(loss_history) >= window:
                moving_avg = np.convolve(loss_history, np.ones(window) / window, mode='valid')
                ax_l.plot(range(window, len(loss_history) + 1), moving_avg,
                          color='red', linewidth=2, label=f'{window}-episode moving avg')
            ax_l.set_xlabel('Episode')
            ax_l.set_ylabel('Loss')
            ax_l.set_title(f'{agent_label} Training Loss')
            ax_l.legend()
            ax_l.grid(True, alpha=0.3)
            plt.tight_layout()
            p = sp.with_stem(sp.stem + '_loss')
            plt.savefig(p, dpi=150)
            plt.close(fig_l)


def plot_evaluation(eval_rewards, save_path=None, stats_path=None,
                    reward_ylim=None, hist_xlim=None,
                    agent_label='DQN'):
    '''Plot DQN evaluation results.'''

    if reward_ylim is None:
        reward_ylim = _dynamic_ylim(eval_rewards)
    if hist_xlim is None:
        hist_xlim = reward_ylim

    episodes = range(1, len(eval_rewards) + 1)

    # Combined subplot figure (1x2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Rewards
    ax = axes[0]
    ax.plot(episodes, eval_rewards, alpha=0.7, color='blue')
    ax.axhline(y=200, color='green', linestyle='--', label='Solved threshold')
    ax.axhline(y=np.mean(eval_rewards), color='red', linestyle='-',
               label=f'Mean: {np.mean(eval_rewards):.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'{agent_label} Rewards')
    ax.set_ylim(reward_ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Reward distribution
    ax = axes[1]
    ax.hist(eval_rewards, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=200, color='green', linestyle='--', label='Solved threshold')
    ax.axvline(x=np.mean(eval_rewards), color='red', linestyle='-',
               label=f'Mean: {np.mean(eval_rewards):.1f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title(f'{agent_label} Reward Distribution')
    ax.set_xlim(hist_xlim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    #  Individual plots
    if save_path:
        sp = Path(save_path)

        # Rewards timeseries
        fig_r, ax_r = plt.subplots(figsize=(10, 6))
        ax_r.plot(episodes, eval_rewards, alpha=0.7, color='blue')
        ax_r.axhline(y=200, color='green', linestyle='--', label='Solved threshold')
        ax_r.axhline(y=np.mean(eval_rewards), color='red', linestyle='-',
                     label=f'Mean: {np.mean(eval_rewards):.1f}')
        ax_r.set_xlabel('Episode')
        ax_r.set_ylabel('Reward')
        ax_r.set_title(f'{agent_label} Rewards')
        ax_r.set_ylim(reward_ylim)
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_rewards')
        plt.savefig(p, dpi=150)
        plt.close(fig_r)

        # Reward distribution
        fig_d, ax_d = plt.subplots(figsize=(10, 6))
        ax_d.hist(eval_rewards, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax_d.axvline(x=200, color='green', linestyle='--', label='Solved threshold')
        ax_d.axvline(x=np.mean(eval_rewards), color='red', linestyle='-',
                     label=f'Mean: {np.mean(eval_rewards):.1f}')
        ax_d.set_xlabel('Reward')
        ax_d.set_ylabel('Count')
        ax_d.set_title(f'{agent_label} Reward Distribution')
        ax_d.set_xlim(hist_xlim)
        ax_d.legend()
        ax_d.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_dist')
        plt.savefig(p, dpi=150)
        plt.close(fig_d)

    # Calculate statistics
    stats = {
        'agent': 'dqn',
        'episodes': len(eval_rewards),
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'min_reward': np.min(eval_rewards),
        'max_reward': np.max(eval_rewards),
        'solved_count': sum(r > 200 for r in eval_rewards),
        'solved_pct': sum(r > 200 for r in eval_rewards) / len(eval_rewards) * 100,
    }

    if stats_path:
        with open(stats_path, 'w') as f:
            f.write(','.join(stats.keys()) + '\n')
            f.write(','.join(str(v) for v in stats.values()) + '\n')

    print('\n' + '=' * 40)
    print('DQN EVALUATION SUMMARY')
    print('=' * 40)
    print(f"Episodes:        {stats['episodes']}")
    print(f"Mean reward:     {stats['mean_reward']:.2f}")
    print(f"Std reward:      {stats['std_reward']:.2f}")
    print(f"Min reward:      {stats['min_reward']:.2f}")
    print(f"Max reward:      {stats['max_reward']:.2f}")
    print(f"Solved (>200):   {stats['solved_count']}/{stats['episodes']} ({stats['solved_pct']:.1f}%)")
    print('=' * 40)

#%% End of Script
