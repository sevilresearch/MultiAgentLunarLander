# -*- coding: utf-8 -*-
# heuristic_lunar_lander.py

'''
Description:
    Executes Lunar Lander using Heuristic agent. Output is an inline demo
    video with option to save and console oberservations ('demo' option) or an
    agent evalution that includes a csv containing agent statistics, a reward
    distribution plot, and console performance summary ('evaluate' option).
'''


#%% Import packages

import numpy as np
import gymnasium as gym
from gymnasium.utils.step_api_compatibility import step_api_compatibility
import matplotlib.pyplot as plt


#%% Constants

# Constants
FPS = 50
VIEWPORT_W = 600
VIEWPORT_H = 400


#%% Functions

def heuristic(s):
    '''
    Assesses state and proximity to goal then returns correction action for
    next step.

    Args:
        s: State vector with 8 elements:
            s[0] - x position
            s[1] - y position
            s[2] - x velocity
            s[3] - y velocity
            s[4] - angle
            s[5] - angular velocity
            s[6] - left leg contact
            s[7] - right leg contact

    Returns:
        Action (0-3)
    '''
    angle_targ = s[0] * 0.5 + s[2] * 1.0
    if angle_targ > 0.4:
        angle_targ = 0.4
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:
        angle_todo = 0
        hover_todo = -(s[3]) * 0.5

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > +0.05:
        a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False,
                          save_video=False, playback_speed=1.0):
    total_reward = 0
    steps = 0
    s, info = env.reset(seed=seed)

    if save_video:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_fps = int(FPS * playback_speed)
        out = cv2.VideoWriter('heuristic_lunar_lander.mp4',
                              fourcc, video_fps, (VIEWPORT_W, VIEWPORT_H))

    while True:
        a = heuristic(s)
        s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        if save_video:
            frame = env.render()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

        if steps % 20 == 0 or terminated or truncated:
            print('observations:', ' '.join([f'{x:+0.2f}' for x in s]))
            print(f'step {steps} total_reward {total_reward:+0.2f}')
        steps += 1

        if terminated or truncated:
            break

    if save_video:
        out.release()
        print('Video saved to heuristic_lunar_lander.mp4')

    if render:
        env.close()

    return total_reward


def evaluate_heuristic(env, num_episodes=100,
                       episode_stats_path='heuristic_episode_stats.csv'):
    rewards = []
    episode_data = []

    for ep in range(num_episodes):
        s, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            a = heuristic(s)
            s, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            total_reward += r
            steps += 1

        rewards.append(total_reward)
        episode_data.append({
            'episode': ep + 1,
            'reward': total_reward,
            'steps': steps,
            'terminated': terminated,
            'truncated': truncated
        })

        if (ep + 1) % 10 == 0:
            print(f'Heuristic episode {ep + 1}/{num_episodes} | Reward: \
                  {total_reward:.2f} | Steps: {steps}')

    # Save per-episode stats to CSV
    with open(episode_stats_path, 'w') as f:
        f.write('episode,reward,steps,terminated,truncated\n')
        for ep_data in episode_data:
            f.write(f'{ep_data["episode"]},{ep_data["reward"]:.2f}, \
                    {ep_data["steps"]},{ep_data["terminated"]}, \
                    {ep_data["truncated"]}\n')
    print(f'Episode stats saved to {episode_stats_path}')

    return rewards

def plot_evaluation(eval_rewards, save_path='heuristic_evaluation.png',
                    stats_path='heuristic_stats.csv', window=50):
    # Episode rewards plot
    _, ax1 = plt.subplots(figsize=(10, 5))
    episodes = range(1, len(eval_rewards) + 1)
    ax1.plot(episodes, eval_rewards, alpha=0.3, color='blue',
             label='Episode reward')

    if len(eval_rewards) >= window:
        moving_avg = np.convolve(eval_rewards, np.ones(window)/window,
                                 mode='valid')
        ax1.plot(range(window, len(eval_rewards) + 1), moving_avg,
                color='red', linewidth=2, label=f'{window}-episode moving avg')

    ax1.axhline(y=200, color='green', linestyle='--', label='Solved threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Heuristic Evaluation Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Evaluation rewards plot saved to {save_path}')

    # Distribution plot
    _, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(eval_rewards, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=200, color='green', linestyle='--', label='Solved threshold')
    ax2.axvline(x=np.mean(eval_rewards), color='red', linestyle='-',
                label=f'Mean: {np.mean(eval_rewards):.1f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Count')
    ax2.set_title('Heuristic Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    dist_save_path = save_path.replace('.png', '_dist.png')
    plt.savefig(dist_save_path, dpi=150)
    plt.show()
    print(f'Distribution plot saved to {dist_save_path}')

    # Calculate statistics
    stats = {
        'agent': 'heuristic',
        'episodes': len(eval_rewards),
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'min_reward': np.min(eval_rewards),
        'max_reward': np.max(eval_rewards),
        'solved_count': sum(r > 200 for r in eval_rewards),
        'solved_pct': sum(r > 200 for r 
                          in eval_rewards) / len(eval_rewards) * 100
    }

    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(','.join(stats.keys()) + '\n')
        f.write(','.join(str(v) for v in stats.values()) + '\n')

    print('\n' + '='*40)
    print('HEURISTIC EVALUATION SUMMARY')
    print('='*40)
    print(f'Episodes:        {stats["episodes"]}')
    print(f'Mean reward:     {stats["mean_reward"]:.2f}')
    print(f'Std reward:      {stats["std_reward"]:.2f}')
    print(f'Min reward:      {stats["min_reward"]:.2f}')
    print(f'Max reward:      {stats["max_reward"]:.2f}')
    print(f'Solved (>200):   {stats["solved_count"]}/{stats["episodes"]} \
          ({stats["solved_pct"]:.1f}%)')
    print('='*40)
    print(f'Stats saved to {stats_path}')




if __name__ == '__main__':

    #%% RUN CONFIGURATION
    MODE = 'demo'           # 'demo' or 'evaluate'
    EVAL_EPISODES = 2000
    SAVE_VIDEO = False
    PLAYBACK_SPEED = 0.5

    # Demo execution
    if MODE == 'demo':
        render_mode = 'rgb_array' if SAVE_VIDEO else 'human'
        env = gym.make('LunarLander-v3', render_mode=render_mode)

        demo_heuristic_lander(
            env,
            render=not SAVE_VIDEO,
            save_video=SAVE_VIDEO,
            playback_speed=PLAYBACK_SPEED
        )
        env.close()

    # Evaluation execution
    elif MODE == 'evaluate':
        env = gym.make('LunarLander-v3')

        print('Evaluating heuristic agent...')
        eval_rewards = evaluate_heuristic(env, num_episodes=EVAL_EPISODES)
        env.close()

        # Plot evaluation results
        plot_evaluation(eval_rewards)

#%% End of Script
