# -*- coding: utf-8 -*-
# multi_agent/utils/plotting.py
# Cooperative training and evaluation plots

#%% Import packages

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from shared_utils.constants import COOP_SUCCESS_THRESHOLD

#%% Constants

# Consistent y-axis limits across all reward plots
REWARD_YLIM = (-500, 500)


#%% Functions

def _dynamic_ylim(rewards, padding=0.1):
    '''Compute y-axis limits with padding from actual data range.'''
    rmin, rmax = np.min(rewards), np.max(rewards)
    span = rmax - rmin if rmax > rmin else 50
    return (rmin - padding * span, rmax + padding * span)

def plot_coop_training(rewards1, rewards2, losses1, losses2, joint_successes,
                       window=50, save_path=None, agent1_label='Agent 1',
                       agent2_label='Agent 2', title_prefix='Cooperative'):
    '''Plot cooperative training curves (2x2 panel).'''

    all_rewards = np.concatenate([rewards1, rewards2])
    ylim = _dynamic_ylim(all_rewards)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: Training rewards
    ax = axes[0, 0]
    episodes = range(1, len(rewards1) + 1)
    ax.plot(episodes, rewards1, alpha=0.2, color='blue', label=agent1_label)
    ax.plot(episodes, rewards2, alpha=0.2, color='orange', label=agent2_label)
    if len(rewards1) >= window:
        avg1 = np.convolve(rewards1, np.ones(window) / window, mode='valid')
        avg2 = np.convolve(rewards2, np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(rewards1) + 1), avg1,
                color='blue', linewidth=2, label=f'{agent1_label} ({window}-ep avg)')
        ax.plot(range(window, len(rewards2) + 1), avg2,
                color='orange', linewidth=2, label=f'{agent2_label} ({window}-ep avg)')
    ax.axhline(y=COOP_SUCCESS_THRESHOLD, color='green', linestyle='--',
               label=f'Threshold ({COOP_SUCCESS_THRESHOLD})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'{title_prefix} Training Rewards')
    ax.set_ylim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Joint success rate
    ax = axes[0, 1]
    if len(joint_successes) >= window:
        rolling = np.convolve(
            [float(s) for s in joint_successes],
            np.ones(window) / window, mode='valid'
        ) * 100
        ax.plot(range(window, len(joint_successes) + 1), rolling, color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Joint Success Rate (%)')
    ax.set_title('Joint Success Rate (Rolling)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Training losses
    ax = axes[1, 0]
    if losses1:
        ax.plot(range(1, len(losses1) + 1), losses1, alpha=0.3, color='blue', label=agent1_label)
    if losses2:
        ax.plot(range(1, len(losses2) + 1), losses2, alpha=0.3, color='orange', label=agent2_label)
    if losses1 and len(losses1) >= window:
        avg_l1 = np.convolve(losses1, np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(losses1) + 1), avg_l1,
                color='blue', linewidth=2, label=f'{agent1_label} ({window}-ep avg)')
    if losses2 and len(losses2) >= window:
        avg_l2 = np.convolve(losses2, np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(losses2) + 1), avg_l2,
                color='orange', linewidth=2, label=f'{agent2_label} ({window}-ep avg)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Epsilon decay
    ax = axes[1, 1]
    eps = 1.0
    eps_history = []
    for _ in range(len(rewards1)):
        eps_history.append(eps)
        eps = max(0.008, eps * 0.998)
    ax.plot(range(1, len(eps_history) + 1), eps_history, color='purple', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    # Individual plots for each subplot
    if save_path:
        sp = Path(save_path)

        # Training rewards
        fig_r, ax_r = plt.subplots(figsize=(10, 6))
        episodes = range(1, len(rewards1) + 1)
        ax_r.plot(episodes, rewards1, alpha=0.2, color='blue', label=agent1_label)
        ax_r.plot(episodes, rewards2, alpha=0.2, color='orange', label=agent2_label)
        if len(rewards1) >= window:
            avg1 = np.convolve(rewards1, np.ones(window) / window, mode='valid')
            avg2 = np.convolve(rewards2, np.ones(window) / window, mode='valid')
            ax_r.plot(range(window, len(rewards1) + 1), avg1,
                      color='blue', linewidth=2, label=f'{agent1_label} ({window}-ep avg)')
            ax_r.plot(range(window, len(rewards2) + 1), avg2,
                      color='orange', linewidth=2, label=f'{agent2_label} ({window}-ep avg)')
        ax_r.axhline(y=COOP_SUCCESS_THRESHOLD, color='green', linestyle='--',
                     label=f'Threshold ({COOP_SUCCESS_THRESHOLD})')
        ax_r.set_xlabel('Episode')
        ax_r.set_ylabel('Reward')
        ax_r.set_title(f'{title_prefix} Training Rewards')
        ax_r.set_ylim(ylim)
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_rewards')
        plt.savefig(p, dpi=150)
        plt.close(fig_r)

        # Joint success rate
        fig_s, ax_s = plt.subplots(figsize=(10, 6))
        if len(joint_successes) >= window:
            rolling = np.convolve(
                [float(s) for s in joint_successes],
                np.ones(window) / window, mode='valid'
            ) * 100
            ax_s.plot(range(window, len(joint_successes) + 1), rolling, color='green', linewidth=2)
        ax_s.set_xlabel('Episode')
        ax_s.set_ylabel('Joint Success Rate (%)')
        ax_s.set_title('Joint Success Rate (Rolling)')
        ax_s.set_ylim(0, 100)
        ax_s.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_success_rate')
        plt.savefig(p, dpi=150)
        plt.close(fig_s)

        # Training losses
        fig_l, ax_l = plt.subplots(figsize=(10, 6))
        if losses1:
            ax_l.plot(range(1, len(losses1) + 1), losses1, alpha=0.3, color='blue', label=agent1_label)
        if losses2:
            ax_l.plot(range(1, len(losses2) + 1), losses2, alpha=0.3, color='orange', label=agent2_label)
        if losses1 and len(losses1) >= window:
            avg_l1 = np.convolve(losses1, np.ones(window) / window, mode='valid')
            ax_l.plot(range(window, len(losses1) + 1), avg_l1,
                      color='blue', linewidth=2, label=f'{agent1_label} ({window}-ep avg)')
        if losses2 and len(losses2) >= window:
            avg_l2 = np.convolve(losses2, np.ones(window) / window, mode='valid')
            ax_l.plot(range(window, len(losses2) + 1), avg_l2,
                      color='orange', linewidth=2, label=f'{agent2_label} ({window}-ep avg)')
        ax_l.set_xlabel('Episode')
        ax_l.set_ylabel('Loss')
        ax_l.set_title('Training Loss')
        ax_l.legend()
        ax_l.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_loss')
        plt.savefig(p, dpi=150)
        plt.close(fig_l)

        # Epsilon decay
        fig_e, ax_e = plt.subplots(figsize=(10, 6))
        eps = 1.0
        eps_history = []
        for _ in range(len(rewards1)):
            eps_history.append(eps)
            eps = max(0.008, eps * 0.998)
        ax_e.plot(range(1, len(eps_history) + 1), eps_history, color='purple', linewidth=2)
        ax_e.set_xlabel('Episode')
        ax_e.set_ylabel('Epsilon')
        ax_e.set_title('Exploration Rate')
        ax_e.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_epsilon')
        plt.savefig(p, dpi=150)
        plt.close(fig_e)


def plot_coop_evaluation(rewards1, rewards2, save_path=None, stats_path=None,
                         agent1_label='Agent 1', agent2_label='Agent 2'):
    '''Plot cooperative evaluation results.'''

    all_rewards = np.concatenate([rewards1, rewards2])
    ylim = _dynamic_ylim(all_rewards)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Episode rewards
    ax = axes[0]
    episodes = range(1, len(rewards1) + 1)
    ax.plot(episodes, rewards1, alpha=0.7, color='blue', label=agent1_label)
    ax.plot(episodes, rewards2, alpha=0.7, color='orange', label=agent2_label)
    ax.axhline(y=COOP_SUCCESS_THRESHOLD, color='green', linestyle='--',
               label=f'Threshold ({COOP_SUCCESS_THRESHOLD})')
    ax.axhline(y=np.mean(rewards1), color='blue', linestyle=':', alpha=0.8,
               label=f'{agent1_label} Mean: {np.mean(rewards1):.1f}')
    ax.axhline(y=np.mean(rewards2), color='orange', linestyle=':', alpha=0.8,
               label=f'{agent2_label} Mean: {np.mean(rewards2):.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Cooperative Evaluation Rewards')
    ax.set_ylim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Agent 1 reward distribution
    ax = axes[1]
    ax.hist(rewards1, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=COOP_SUCCESS_THRESHOLD, color='green', linestyle='--',
               label=f'Threshold ({COOP_SUCCESS_THRESHOLD})')
    ax.axvline(x=np.mean(rewards1), color='red', linestyle='-',
               label=f'Mean: {np.mean(rewards1):.1f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title(f'{agent1_label} Reward Distribution')
    ax.set_xlim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Agent 2 reward distribution
    ax = axes[2]
    ax.hist(rewards2, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(x=COOP_SUCCESS_THRESHOLD, color='green', linestyle='--',
               label=f'Threshold ({COOP_SUCCESS_THRESHOLD})')
    ax.axvline(x=np.mean(rewards2), color='red', linestyle='-',
               label=f'Mean: {np.mean(rewards2):.1f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title(f'{agent2_label} Reward Distribution')
    ax.set_xlim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    # Individual plots for each subplot
    if save_path:
        sp = Path(save_path)

        # Evaluation rewards
        fig_r, ax_r = plt.subplots(figsize=(10, 6))
        episodes = range(1, len(rewards1) + 1)
        ax_r.plot(episodes, rewards1, alpha=0.7, color='blue', label=agent1_label)
        ax_r.plot(episodes, rewards2, alpha=0.7, color='orange', label=agent2_label)
        ax_r.axhline(y=COOP_SUCCESS_THRESHOLD, color='green', linestyle='--',
                     label=f'Threshold ({COOP_SUCCESS_THRESHOLD})')
        ax_r.axhline(y=np.mean(rewards1), color='blue', linestyle=':', alpha=0.8,
                     label=f'{agent1_label} Mean: {np.mean(rewards1):.1f}')
        ax_r.axhline(y=np.mean(rewards2), color='orange', linestyle=':', alpha=0.8,
                     label=f'{agent2_label} Mean: {np.mean(rewards2):.1f}')
        ax_r.set_xlabel('Episode')
        ax_r.set_ylabel('Reward')
        ax_r.set_title('Cooperative Evaluation Rewards')
        ax_r.set_ylim(ylim)
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_rewards')
        plt.savefig(p, dpi=150)
        plt.close(fig_r)

        # Agent 1 reward distribution
        fig_d1, ax_d1 = plt.subplots(figsize=(10, 6))
        ax_d1.hist(rewards1, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax_d1.axvline(x=COOP_SUCCESS_THRESHOLD, color='green', linestyle='--',
                      label=f'Threshold ({COOP_SUCCESS_THRESHOLD})')
        ax_d1.axvline(x=np.mean(rewards1), color='red', linestyle='-',
                      label=f'Mean: {np.mean(rewards1):.1f}')
        ax_d1.set_xlabel('Reward')
        ax_d1.set_ylabel('Count')
        ax_d1.set_title(f'{agent1_label} Reward Distribution')
        ax_d1.set_xlim(ylim)
        ax_d1.legend()
        ax_d1.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_agent1_dist')
        plt.savefig(p, dpi=150)
        plt.close(fig_d1)

        # Agent 2 reward distribution
        fig_d2, ax_d2 = plt.subplots(figsize=(10, 6))
        ax_d2.hist(rewards2, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax_d2.axvline(x=COOP_SUCCESS_THRESHOLD, color='green', linestyle='--',
                      label=f'Threshold ({COOP_SUCCESS_THRESHOLD})')
        ax_d2.axvline(x=np.mean(rewards2), color='red', linestyle='-',
                      label=f'Mean: {np.mean(rewards2):.1f}')
        ax_d2.set_xlabel('Reward')
        ax_d2.set_ylabel('Count')
        ax_d2.set_title(f'{agent2_label} Reward Distribution')
        ax_d2.set_xlim(ylim)
        ax_d2.legend()
        ax_d2.grid(True, alpha=0.3)
        plt.tight_layout()
        p = sp.with_stem(sp.stem + '_agent2_dist')
        plt.savefig(p, dpi=150)
        plt.close(fig_d2)

    # Save statistics
    joint_successes = sum(
        1 for r1, r2 in zip(rewards1, rewards2)
        if r1 > COOP_SUCCESS_THRESHOLD and r2 > COOP_SUCCESS_THRESHOLD
    )
    stats = {
        'episodes': len(rewards1),
        'agent1_mean': np.mean(rewards1),
        'agent1_std': np.std(rewards1),
        'agent1_min': np.min(rewards1),
        'agent1_max': np.max(rewards1),
        'agent1_solved_pct': sum(r > COOP_SUCCESS_THRESHOLD for r in rewards1) / len(rewards1) * 100,
        'agent2_mean': np.mean(rewards2),
        'agent2_std': np.std(rewards2),
        'agent2_min': np.min(rewards2),
        'agent2_max': np.max(rewards2),
        'agent2_solved_pct': sum(r > COOP_SUCCESS_THRESHOLD for r in rewards2) / len(rewards2) * 100,
        'joint_success_pct': joint_successes / len(rewards1) * 100,
    }

    if stats_path:
        with open(stats_path, 'w') as f:
            f.write(','.join(stats.keys()) + '\n')
            f.write(','.join(str(v) for v in stats.values()) + '\n')

    print('\n' + '=' * 50)
    print('COOPERATIVE EVALUATION SUMMARY')
    print('=' * 50)
    print(f"Episodes:            {stats['episodes']}")
    print(f"{agent1_label} mean reward: {stats['agent1_mean']:.2f} +/- {stats['agent1_std']:.2f}")
    print(f"{agent2_label} mean reward: {stats['agent2_mean']:.2f} +/- {stats['agent2_std']:.2f}")
    print(f"{agent1_label} solved:      {stats['agent1_solved_pct']:.1f}%")
    print(f"{agent2_label} solved:      {stats['agent2_solved_pct']:.1f}%")
    print(f"Joint success:       {stats['joint_success_pct']:.1f}%")
    print('=' * 50)

#%% End of Script